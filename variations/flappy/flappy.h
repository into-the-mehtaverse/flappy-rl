/* Flappy: single-agent Flappy Bird-style env. C + raylib. */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define MAX_PIPES 5
#define OBS_DIM 5
#define BIRD_X_RATIO 0.2f
#define PIPE_WIDTH_RATIO 0.15f
#define BIRD_RADIUS_RATIO 0.025f  /* was 0.03; smaller = more margin through gap */
#define GAP_HEIGHT_RATIO 0.28f
/* Continuous curriculum: difficulty float in [0, 1] controls gap distribution.
 *   d  0.00–0.25 : range widens from center-only to full [0.25, 0.75]
 *   d  0.25–0.55 : full range + increasing extreme-bias (peaks ~45 %)
 *   d  0.55–0.85 : full range + decreasing extreme-bias
 *   d  0.85–1.00 : pure uniform [0.25, 0.75]  (matches eval)
 */
#define PIPE_SPEED_RATIO 0.006f  /* slower pipes so bird has more time to align; was 0.012 */
#define FLAP_VEL 0.02f   /* upward velocity per flap; lower = finer control, less overshoot */
#define GRAVITY 0.0018f
#define PIPE_SPACING_RATIO 0.45f
/* Sparse reward: +1 pipe pass, -1 death. No shaping. */

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float difficulty;  /* curriculum difficulty (0.0–1.0) for dashboard */
    float n;
} Log;

typedef struct {
    float x;
    float gap_center_y;
    float gap_height;
    int scored;
} Pipe;

typedef struct Client {
    Texture2D bird;
    Texture2D pipe;
} Client;

typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    int width;
    int height;
    float gravity;
    float flap_velocity;
    float pipe_speed;
    float pipe_spacing;
    float gap_height;
    int max_steps;

    float bird_y;
    float bird_vy;
    Pipe pipes[MAX_PIPES];
    int num_pipes;
    int score;
    int step_count;
    float curriculum_difficulty;  /* 0.0 = fixed center, 1.0 = full uniform */
    Client* client;
} Flappy;

static void add_log(Flappy* env) {
    env->log.perf = env->score > 0 ? 1.0f : 0.0f;
    env->log.score = (float)env->score;
    env->log.episode_length = (float)env->step_count;
    env->log.difficulty = env->curriculum_difficulty;
    env->log.n += 1.0f;
}

void init(Flappy* env) {
    env->gravity = GRAVITY;
    env->flap_velocity = FLAP_VEL;
    env->pipe_speed = (float)env->width * PIPE_SPEED_RATIO;
    env->pipe_spacing = PIPE_SPACING_RATIO;
    env->gap_height = GAP_HEIGHT_RATIO;
    if (env->max_steps <= 0) env->max_steps = 5000;
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Only sets gap and scored; caller sets x. Gap distribution is a smooth
 * function of curriculum_difficulty (0–1): range widens, extreme-bias
 * ramps up then back down, ending on pure uniform (= eval distribution). */
static void spawn_pipe(Flappy* env, int idx) {
    float d = env->curriculum_difficulty;

    /* 1. Range expansion: d 0→0.25 widens half-range from 0 to 0.25 */
    float half_range = (d < 0.25f) ? d : 0.25f;
    float gap_min = 0.5f - half_range;
    float gap_max = 0.5f + half_range;

    /* 2. Extreme-bias: smooth hump peaking at d=0.55 (~45 %), zero
     *    outside [0.25, 0.85] so training ends on pure uniform. */
    float extreme_prob = 0.0f;
    if (d > 0.25f && d < 0.85f) {
        float peak_d = 0.55f;
        float t;
        if (d <= peak_d)
            t = (d - 0.25f) / (peak_d - 0.25f);   /* 0 → 1 rising  */
        else
            t = 1.0f - (d - peak_d) / (0.85f - peak_d); /* 1 → 0 falling */
        extreme_prob = t * 0.45f;  /* peak 45 % extreme sampling */
    }

    /* 3. Sample gap center */
    float r = (float)(rand() % 1000) / 1000.0f;
    if (r < extreme_prob) {
        /* Extreme band: [0.25, 0.35] or [0.65, 0.75] */
        if (rand() % 2 == 0)
            env->pipes[idx].gap_center_y = 0.25f + (float)(rand() % 11) / 100.0f;
        else
            env->pipes[idx].gap_center_y = 0.65f + (float)(rand() % 11) / 100.0f;
    } else {
        /* Uniform within current range */
        int steps = (int)((gap_max - gap_min) * 100.0f + 0.5f);
        if (steps <= 0)
            env->pipes[idx].gap_center_y = 0.5f;
        else
            env->pipes[idx].gap_center_y = gap_min + (float)(rand() % (steps + 1)) / 100.0f;
    }

    env->pipes[idx].gap_height = env->gap_height;
    env->pipes[idx].scored = 0;
}

/* 5-dim obs: bird_y, bird_vy, dist_to_pipe, gap_center, gap_height */
void compute_observations(Flappy* env) {
    float* o = env->observations;
    o[0] = clampf(env->bird_y, 0.0f, 1.0f);
    o[1] = clampf(env->bird_vy / 0.1f, -1.0f, 1.0f);
    float bird_x = (float)env->width * BIRD_X_RATIO;
    float pw = (float)env->width * PIPE_WIDTH_RATIO;
    int next = -1;
    float best_x = 1e9f;
    for (int i = 0; i < env->num_pipes; i++) {
        if (env->pipes[i].x + pw > bird_x && env->pipes[i].x < best_x) {
            best_x = env->pipes[i].x;
            next = i;
        }
    }
    if (next >= 0) {
        float dx = env->pipes[next].x - bird_x;
        o[2] = clampf(dx / (float)env->width, 0.0f, 1.0f);
        o[3] = env->pipes[next].gap_center_y;
        o[4] = env->pipes[next].gap_height;
    } else {
        o[2] = 1.0f;
        o[3] = 0.5f;
        o[4] = env->gap_height;
    }
}

static int collides(Flappy* env, float bx, float by, float br) {
    float pw = env->width * PIPE_WIDTH_RATIO;
    float ph_top = env->height * 2.0f;
    for (int i = 0; i < env->num_pipes; i++) {
        float px = env->pipes[i].x;
        if (px + pw < bx - br || px > bx + br) continue;
        float gap_c = env->pipes[i].gap_center_y * (float)env->height;
        float gap_h = env->pipes[i].gap_height * (float)env->height;
        float top_bottom = gap_c - gap_h * 0.5f;
        float bottom_top = gap_c + gap_h * 0.5f;
        if (by - br < top_bottom || by + br > bottom_top)
            return 1;
    }
    return 0;
}

void c_reset(Flappy* env, float difficulty) {
    env->curriculum_difficulty = difficulty;
    env->log.episode_return = 0.0f;
    env->bird_y = 0.5f;
    env->bird_vy = 0.0f;
    env->score = 0;
    env->step_count = 0;
    env->num_pipes = 3;
    float start_x = (float)env->width * 0.5f;
    for (int i = 0; i < env->num_pipes; i++) {
        env->pipes[i].x = start_x + (float)i * env->width * env->pipe_spacing;
        spawn_pipe(env, i);
    }
    compute_observations(env);
}

void c_step(Flappy* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->step_count++;

    /* Physics */
    int a = env->actions[0];
    if (a == 1)
        env->bird_vy = -env->flap_velocity;
    env->bird_vy += env->gravity;
    env->bird_y += env->bird_vy;
    env->bird_y = clampf(env->bird_y, 0.0f, 1.0f);

    /* Collision: ceiling / floor */
    float by_px = env->bird_y * (float)env->height;
    float bx_px = (float)env->width * BIRD_X_RATIO;
    float br = (float)env->height * BIRD_RADIUS_RATIO;
    if (by_px - br <= 0.0f || by_px + br >= (float)env->height) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env, env->curriculum_difficulty);
        return;
    }
    /* Collision: pipes */
    if (collides(env, bx_px, by_px, br)) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env, env->curriculum_difficulty);
        return;
    }

    /* Scoring: +1 per pipe passed */
    float pw = env->width * PIPE_WIDTH_RATIO;
    for (int i = 0; i < env->num_pipes; i++) {
        if (!env->pipes[i].scored && env->pipes[i].x + pw < bx_px) {
            env->pipes[i].scored = 1;
            env->rewards[0] += 1.0f;
            env->score++;
        }
    }

    /* Move pipes & recycle */
    for (int i = 0; i < env->num_pipes; i++)
        env->pipes[i].x -= env->pipe_speed;

    int leftmost = 0;
    for (int i = 1; i < env->num_pipes; i++)
        if (env->pipes[i].x < env->pipes[leftmost].x) leftmost = i;
    if (env->pipes[leftmost].x + pw < 0) {
        float rightmost = env->pipes[0].x;
        for (int i = 1; i < env->num_pipes; i++)
            if (env->pipes[i].x > rightmost) rightmost = env->pipes[i].x;
        env->pipes[leftmost].x = rightmost + (float)env->width * env->pipe_spacing;
        spawn_pipe(env, leftmost);
    }

    /* Truncation */
    if (env->step_count >= env->max_steps) {
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env, env->curriculum_difficulty);
        return;
    }
    env->log.episode_return += env->rewards[0];
    compute_observations(env);
}

void c_render(Flappy* env) {
    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        InitWindow(env->width, env->height, "Flappy");
        SetTargetFPS(60);
        env->client->bird = LoadTexture("resources/flappy/bird.png");
        env->client->pipe = LoadTexture("resources/flappy/pipe.png");
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    Client* c = env->client;
    BeginDrawing();
    ClearBackground((Color){113, 197, 207, 255});

    float pw = (float)env->width * PIPE_WIDTH_RATIO;
    float gap_c, gap_h, top_bottom, bottom_top;
    for (int i = 0; i < env->num_pipes; i++) {
        gap_c = env->pipes[i].gap_center_y * (float)env->height;
        gap_h = env->pipes[i].gap_height * (float)env->height;
        top_bottom = gap_c - gap_h * 0.5f;
        bottom_top = gap_c + gap_h * 0.5f;
        DrawTexturePro(c->pipe,
            (Rectangle){0, 0, (float)c->pipe.width, (float)c->pipe.height},
            (Rectangle){env->pipes[i].x, 0, pw, top_bottom},
            (Vector2){0, 0}, 0, WHITE);
        DrawTexturePro(c->pipe,
            (Rectangle){0, 0, (float)c->pipe.width, (float)c->pipe.height},
            (Rectangle){env->pipes[i].x, bottom_top, pw, (float)env->height - bottom_top},
            (Vector2){0, 0}, 0, WHITE);
    }

    float by = env->bird_y * (float)env->height;
    float bx = (float)env->width * BIRD_X_RATIO;
    float br = (float)env->height * BIRD_RADIUS_RATIO * 2.0f;
    DrawTexturePro(c->bird,
        (Rectangle){0, 0, (float)c->bird.width, (float)c->bird.height},
        (Rectangle){bx - br, by - br, br * 2, br * 2},
        (Vector2){br, br}, 0, WHITE);

    DrawText(TextFormat("Score: %d", env->score), 10, 10, 20, DARKGRAY);
    EndDrawing();
}

void c_close(Flappy* env) {
    if (env->client) {
        UnloadTexture(env->client->bird);
        UnloadTexture(env->client->pipe);
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}
