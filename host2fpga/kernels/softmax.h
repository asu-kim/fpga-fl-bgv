#ifndef SOFTMAX_H
#define SOFTMAX_H

// taylor approximation
static float exp_approx(float x) {
    float sum = 1.0f;
    float term = 1.0f;
    term *= x;        sum += term;         // x^1/1!
    term *= x * 0.5f; sum += term;         // x^2/2!
    term *= x / 3.0f; sum += term;         // x^3/3!
    term *= x / 4.0f; sum += term;         // x^4/4!
    term *= x / 5.0f; sum += term;         // x^5/5!
    term *= x / 6.0f; sum += term;         // x^6/6!
    return sum;
}

template<int N>
void softmax(
        const float logits[N],
        float probs[N]
        ) {
    float max_logit = logits[0];
    for(int i=1; i<N; ++i) {
        if(logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum = 0.0f;
    for(int i=0; i<N; ++i) {
        float diff = (logits[i] - max_logit);
        float e = exp_approx(diff);
        probs[i] = e;
        sum += e;
    }

    if(sum == 0.0f) {
        float uniform = 1.0f / N;
        for(int i=0; i<N; ++i) {
            probs[i] = uniform;
        }
    } else {
        for(int i=0; i<N; ++i) {
            probs[i] /= sum;
        }
    }
}


#endif
