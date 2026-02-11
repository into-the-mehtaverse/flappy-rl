/* Flappy: single-agent Flappy Bird-style env. C + raylib. */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

#define MAX_PIPES 5
#define OBS_DIM 9
#define BIRD_X_RATIO 0.2f
#define PIPE_WIDTH_RATIO 0.15f
#define BIRD_RADIUS_RATIO 0.025f  /* was 0.03; smaller = more margin through gap */
#define GAP_HEIGHT_RATIO 0.28f
#define FIXED_GAP_DEBUG 0       /* 1 = all gaps at same height (debug); 0 = random gap center 0.25–0.75 */
#define FIXED_GAP_CENTER_Y 0.5f
#define BIAS_HARD_GAPS 0        /* 1 = sample gap from extremes [0.25,0.35] and [0.65,0.75] only (training); 0 = uniform [0.25,0.75] */
#define PIPE_SPEED_RATIO 0.006f  /* slower pipes so bird has more time to align; was 0.012 */
#define FLAP_VEL 0.02f   /* upward velocity per flap; lower = finer control, less overshoot */
#define GRAVITY 0.0018f
#define PIPE_SPACING_RATIO 0.45f
#define SURVIVAL_BONUS 0.01f  /* small reward per step alive so policy learns to flap to avoid ground */
#define IN_GAP_BONUS 0.02f    /* reward when bird is inside the gap; scaled by distance to pipe */
#define ALIGNMENT_BONUS 0.008f   /* tiny reward for being near gap center (before entering); encourages lining up early */
#define ALIGNMENT_TOLERANCE 0.2f /* normalized y distance over which alignment bonus decays (wider than gap) */
#define STREAK_BONUS 0.1f       /* extra reward per pipe already passed (1st=1.0, 2nd=1.1, 3rd=1.2, ...) */
#define FLAP_PENALTY 0.001f    /* tiny cost per flap to discourage unnecessary flapping (e.g. when already high) */

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
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
    Client* client;
} Flappy;

static void add_log(Flappy* env) {
    env->log.perf = env->score > 0 ? 1.0f : 0.0f;
    env->log.score = (float)env->score;
    /* episode_return already accumulated each step in c_step */
    env->log.episode_length = (float)env->step_count;
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

/* Only sets gap and scored; caller sets x (recycle uses rightmost+spacing, reset sets start_x+i*spacing) */
static void spawn_pipe(Flappy* env, int idx) {
#if FIXED_GAP_DEBUG
    env->pipes[idx].gap_center_y = FIXED_GAP_CENTER_Y;
#elif BIAS_HARD_GAPS
    /* Sample only from extreme bands so policy sees more hard cases */
    if (rand() % 2 == 0)
        env->pipes[idx].gap_center_y = 0.25f + (float)(rand() % 11) / 100.0f;  /* [0.25, 0.35] */
    else
        env->pipes[idx].gap_center_y = 0.65f + (float)(rand() % 11) / 100.0f;  /* [0.65, 0.75] */
#else
    env->pipes[idx].gap_center_y = 0.25f + (float)(rand() % 50) / 100.0f;
#endif
    env->pipes[idx].gap_height = env->gap_height;
    env->pipes[idx].scored = 0;
}

void compute_observations(Flappy* env) {
    float* o = env->observations;
    o[0] = clampf(env->bird_y, 0.0f, 1.0f);
    o[1] = clampf(env->bird_vy / 0.1f, -1.0f, 1.0f);
    float bird_x = (float)env->width * BIRD_X_RATIO;
    float pw = (float)env->width * PIPE_WIDTH_RATIO;
    /* next = pipe in front of bird that is closest (leftmost in front), not first in array order (array order changes after recycling) */
    int next = -1;
    float best_x = 1e9f;
    for (int i = 0; i < env->num_pipes; i++) {
        if (env->pipes[i].x + pw > bird_x && env->pipes[i].x < best_x) {
            best_x = env->pipes[i].x;
            next = i;
        }
    }
    float dist_norm = 1.0f;
    float gap_center = 0.5f;
    float gap_h = env->gap_height;
    if (next >= 0) {
        float dx = env->pipes[next].x - bird_x;
        dist_norm = clampf(dx / (float)env->width, 0.0f, 1.0f);
        gap_center = env->pipes[next].gap_center_y;
        gap_h = env->pipes[next].gap_height;
    }
    o[2] = dist_norm;
    o[3] = gap_center;
    o[4] = gap_h;
    o[5] = (next >= 0) ? 1.0f : 0.0f;
    /* o[6]: signed gap error = gap_center - bird_y. Positive = bird below gap (flap more), negative = bird above gap (cool it) */
    o[6] = (next >= 0) ? clampf(gap_center - env->bird_y, -1.0f, 1.0f) : 0.0f;
    /* o[7]: clearance from top of gap (in half-gap units). Positive = bird below top edge (safe), negative = above top (danger) */
    /* o[8]: clearance from bottom of gap. Negative = bird above bottom (safe), positive = below bottom (danger) */
    if (next >= 0) {
        float half = gap_h * 0.5f;
        float top_edge = gap_center - half;
        float bottom_edge = gap_center + half;
        if (half > 1e-6f) {
            o[7] = clampf((top_edge - env->bird_y) / half, -1.0f, 1.0f);
            o[8] = clampf((env->bird_y - bottom_edge) / half, -1.0f, 1.0f);
        } else {
            o[7] = 0.0f;
            o[8] = 0.0f;
        }
    } else {
        o[7] = 0.0f;
        o[8] = 0.0f;
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

void c_reset(Flappy* env) {
    env->log.episode_return = 0.0f;
    env->bird_y = 0.5f;
    env->bird_vy = 0.0f;
    env->score = 0;
    env->step_count = 0;
    env->num_pipes = 3;
    /* first pipe at half the previous distance so agent learns to react quickly (like between pipes) */
    float start_x = (float)env->width * 0.5f;  /* was 0.8: bird at 0.2, so distance 0.6→0.3 */
    for (int i = 0; i < env->num_pipes; i++) {
        env->pipes[i].x = start_x + (float)i * env->width * env->pipe_spacing;
#if FIXED_GAP_DEBUG
        env->pipes[i].gap_center_y = FIXED_GAP_CENTER_Y;
#elif BIAS_HARD_GAPS
        if (rand() % 2 == 0)
            env->pipes[i].gap_center_y = 0.25f + (float)(rand() % 11) / 100.0f;
        else
            env->pipes[i].gap_center_y = 0.65f + (float)(rand() % 11) / 100.0f;
#else
        env->pipes[i].gap_center_y = 0.25f + (float)(rand() % 50) / 100.0f;
#endif
        env->pipes[i].gap_height = env->gap_height;
        env->pipes[i].scored = 0;
    }
    compute_observations(env);
}

void c_step(Flappy* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->step_count++;

    int a = env->actions[0];
    if (a == 1) {
        env->bird_vy = -env->flap_velocity;
        env->rewards[0] -= FLAP_PENALTY;
    }
    env->bird_vy += env->gravity;
    env->bird_y += env->bird_vy;
    env->bird_y = clampf(env->bird_y, 0.0f, 1.0f);

    float by_px = env->bird_y * (float)env->height;
    float bx_px = (float)env->width * BIRD_X_RATIO;
    float br = (float)env->height * BIRD_RADIUS_RATIO;
    if (by_px - br <= 0.0f || by_px + br >= (float)env->height) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }
    if (collides(env, bx_px, by_px, br)) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
        return;
    }

    float pw = env->width * PIPE_WIDTH_RATIO;
    for (int i = 0; i < env->num_pipes; i++) {
        if (!env->pipes[i].scored && env->pipes[i].x + pw < bx_px) {
            env->pipes[i].scored = 1;
            env->rewards[0] += 1.0f + STREAK_BONUS * (float)env->score;
            env->score++;
        }
    }

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

    env->rewards[0] += SURVIVAL_BONUS;

    /* in-gap bonus only (scaled by distance to pipe); no penalty for being out */
    {
        float bird_x = (float)env->width * BIRD_X_RATIO;
        int next = -1;
        float best_x = 1e9f;
        for (int i = 0; i < env->num_pipes; i++) {
            if (env->pipes[i].x + pw > bird_x && env->pipes[i].x < best_x) {
                best_x = env->pipes[i].x;
                next = i;
            }
        }
        if (next >= 0) {
            float gap_center = env->pipes[next].gap_center_y;
            float gap_h = env->pipes[next].gap_height;
            float half = gap_h * 0.5f;
            float dx = env->pipes[next].x - bird_x;
            float dist_norm = clampf(dx / (float)env->width, 0.0f, 1.0f);
            float scale = 1.0f - dist_norm;

            if (env->bird_y >= gap_center - half && env->bird_y <= gap_center + half) {
                env->rewards[0] += scale * IN_GAP_BONUS;
            }
            /* alignment: small reward for being near gap center even before entering (encourages lining up early) */
            {
                float align_err = fabsf(env->bird_y - gap_center);
                float align_scale = 1.0f - clampf(align_err / ALIGNMENT_TOLERANCE, 0.0f, 1.0f);
                env->rewards[0] += ALIGNMENT_BONUS * align_scale;
            }
        }
    }

    if (env->step_count >= env->max_steps) {
        env->terminals[0] = 1;
        env->log.episode_return += env->rewards[0];
        add_log(env);
        c_reset(env);
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
