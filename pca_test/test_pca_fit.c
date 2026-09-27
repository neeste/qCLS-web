/*
 * PCA post-hoc recovery test.
 *
 * This harness treats each profile in pca_test/ground_truth.csv as the
 * listener's true set of 100 category boundaries. It generates synthetic
 * responses at five levels around every boundary and passes those trials to
 * the production qcls_pca_fit_report() implementation. The recovered profile
 * is compared with the ground truth in dB; make test-pca-fit fails if the
 * aggregate RMSE exceeds 8 dB. Responses use the browser's 0, 5, ..., 50 CU
 * scale so this also guards the production likelihood encoding.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../qcls_core.c"

#define MAX_LISTENERS 148
#define TRIALS_PER_PROFILE (PCA_NFREQS * PCA_BOUNDARIES * 5)

static float truth[MAX_LISTENERS][PCA_NFREQS][PCA_BOUNDARIES];

static int load_ground_truth(const char* path, int* listener_count) {
    FILE* file = fopen(path, "r");
    if (!file) return 0;

    char line[4096];
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return 0;
    }

    int max_listener = 0;
    while (fgets(line, sizeof(line), file)) {
        int listener = 0;
        int boundary = 0;
        float values[PCA_NFREQS];
        int parsed = sscanf(line,
            "%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
            &listener, &boundary,
            &values[0], &values[1], &values[2], &values[3], &values[4],
            &values[5], &values[6], &values[7], &values[8], &values[9]);
        if (parsed != 12 || listener < 1 || listener > MAX_LISTENERS ||
            boundary < 1 || boundary > PCA_BOUNDARIES) {
            fclose(file);
            return 0;
        }
        for (int frequency = 0; frequency < PCA_NFREQS; frequency++) {
            truth[listener - 1][frequency][boundary - 1] = values[frequency];
        }
        if (listener > max_listener) max_listener = listener;
    }

    fclose(file);
    *listener_count = max_listener;
    return 1;
}

static int category_response(int listener, int frequency, float level) {
    int category = 0;
    for (int boundary = 0; boundary < PCA_BOUNDARIES; boundary++) {
        if (level >= truth[listener][frequency][boundary]) category++;
    }
    return category * 5;
}

static int build_trials(int listener, float* frequencies, float* levels, float* responses) {
    static const float offsets[5] = {-8.0f, -3.0f, 0.0f, 3.0f, 8.0f};
    int trial = 0;
    for (int frequency = 0; frequency < PCA_NFREQS; frequency++) {
        for (int boundary = 0; boundary < PCA_BOUNDARIES; boundary++) {
            for (int offset = 0; offset < 5; offset++) {
                float level = truth[listener][frequency][boundary] + offsets[offset];
                frequencies[trial] = 250.0f;
                if (frequency == 1) frequencies[trial] = 500.0f;
                if (frequency == 2) frequencies[trial] = 750.0f;
                if (frequency == 3) frequencies[trial] = 1000.0f;
                if (frequency == 4) frequencies[trial] = 1500.0f;
                if (frequency == 5) frequencies[trial] = 2000.0f;
                if (frequency == 6) frequencies[trial] = 3000.0f;
                if (frequency == 7) frequencies[trial] = 4000.0f;
                if (frequency == 8) frequencies[trial] = 6000.0f;
                if (frequency == 9) frequencies[trial] = 8000.0f;
                levels[trial] = level;
                responses[trial] = (float)category_response(listener, frequency, level);
                trial++;
            }
        }
    }
    return trial;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "pca_test/ground_truth.csv";
    int limit = (argc > 2) ? atoi(argv[2]) : MAX_LISTENERS;
    if (limit < 1 || limit > MAX_LISTENERS) limit = MAX_LISTENERS;

    int listener_count = 0;
    if (!load_ground_truth(path, &listener_count)) {
        fprintf(stderr, "could not load ground truth: %s\n", path);
        return 2;
    }
    if (limit > listener_count) limit = listener_count;

    float* frequencies = malloc(sizeof(float) * TRIALS_PER_PROFILE);
    float* levels = malloc(sizeof(float) * TRIALS_PER_PROFILE);
    float* responses = malloc(sizeof(float) * TRIALS_PER_PROFILE);
    float* estimate = malloc(sizeof(float) * PCA_PARAMS);
    if (!frequencies || !levels || !responses || !estimate) return 2;

    srand(1);
    double sum_squared_error = 0.0;
    double sum_error = 0.0;
    double max_listener_rmse = 0.0;
    int total_values = 0;
    for (int listener = 0; listener < limit; listener++) {
        int trials = build_trials(listener, frequencies, levels, responses);
        if (!qcls_pca_fit_report(frequencies, levels, responses, trials, estimate)) {
            fprintf(stderr, "fit failed for listener %d\n", listener + 1);
            return 1;
        }

        double listener_squared_error = 0.0;
        for (int frequency = 0; frequency < PCA_NFREQS; frequency++) {
            for (int boundary = 0; boundary < PCA_BOUNDARIES; boundary++) {
                int index = frequency * PCA_BOUNDARIES + boundary;
                double error = estimate[index] - truth[listener][frequency][boundary];
                listener_squared_error += error * error;
                sum_squared_error += error * error;
                sum_error += error;
                total_values++;
            }
        }
         double listener_rmse = sqrt(listener_squared_error / PCA_PARAMS);
         if (listener_rmse > max_listener_rmse) max_listener_rmse = listener_rmse;
         printf("listener %d RMSE %.3f dB\n", listener + 1, listener_rmse);
    }

        double aggregate_rmse = sqrt(sum_squared_error / total_values);
        printf("profiles %d, values %d, RMSE %.3f dB, mean bias %.3f dB, max listener RMSE %.3f dB\n",
           limit, total_values,
            aggregate_rmse, sum_error / total_values, max_listener_rmse);
    free(frequencies);
    free(levels);
    free(responses);
    free(estimate);
    return aggregate_rmse <= 8.0 ? 0 : 1;
}
