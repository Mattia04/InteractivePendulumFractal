typedef struct {
    float q1;
    float q2;
    float p1;
    float p2;
} Pendulum;


// Derivative function with proper address spaces
void derivatives(__private const Pendulum* p, __private Pendulum* dpdt) {
    const float g = 9.806f;  // gravity

    // Intermediate calculations
    float cos_1 = cos(p->q1 - p->q2);
    float cos_2 = cos_1 * cos_1;
    float sin_1 = sin(p->q1 - p->q2);
    float denom = (2.f - cos_2);
    float denom_2 = denom * denom;
    float prod = p->p1 * p->p2;
    float numer = sin_1 * ((p->p1 * p->p1 + 2.f * p->p2 * p->p2) * cos_1 - prod * (2.f + cos_2));
    float f2 = numer / denom_2;

    // Angular velocity derivatives
    dpdt->q1 = (-p->p1 + p->p2 * cos_1) / denom;
    dpdt->q2 = (-2.f * p->p2 + p->p1 * cos_1) / denom;

    // Momentum derivatives
    dpdt->p1 = -f2 + 2.f * g * sin(p->q1);
    dpdt->p2 = +f2 + g * sin(p->q2);
}


__kernel void flip_time_simulation(
    __global const Pendulum* pendulums,
    __global float* flip_times,
    float step_size,
    float total_time,
    int N
) {
    int i = get_global_id(0);
    if (i >= N) return;

    Pendulum current = pendulums[i];
    float t = 0.0f;
    float flip_time = -1.0f;

    // remove non flipping zone
    if (2 * cos(current.q1) + cos(current.q2) > 1)
        t = total_time;

    while (t < total_time) {
        Pendulum k1, k2, k3, k4;
        Pendulum temp;

        // k1
        derivatives(&current, &k1);

        // k2
        temp.q1 = current.q1 + 0.5f * step_size * k1.q1;
        temp.q2 = current.q2 + 0.5f * step_size * k1.q2;
        temp.p1 = current.p1 + 0.5f * step_size * k1.p1;
        temp.p2 = current.p2 + 0.5f * step_size * k1.p2;
        derivatives(&temp, &k2);

        // k3
        temp.q1 = current.q1 + 0.5f * step_size * k2.q1;
        temp.q2 = current.q2 + 0.5f * step_size * k2.q2;
        temp.p1 = current.p1 + 0.5f * step_size * k2.p1;
        temp.p2 = current.p2 + 0.5f * step_size * k2.p2;
        derivatives(&temp, &k3);

        // k4
        temp.q1 = current.q1 + step_size * k3.q1;
        temp.q2 = current.q2 + step_size * k3.q2;
        temp.p1 = current.p1 + step_size * k3.p1;
        temp.p2 = current.p2 + step_size * k3.p2;
        derivatives(&temp, &k4);

        // Update state
        current.q1 += step_size / 6.0f * (k1.q1 + 2.0f*k2.q1 + 2.0f*k3.q1 + k4.q1);
        current.q2 += step_size / 6.0f * (k1.q2 + 2.0f*k2.q2 + 2.0f*k3.q2 + k4.q2);
        current.p1 += step_size / 6.0f * (k1.p1 + 2.0f*k2.p1 + 2.0f*k3.p1 + k4.p1);
        current.p2 += step_size / 6.0f * (k1.p2 + 2.0f*k2.p2 + 2.0f*k3.p2 + k4.p2);

        // Check for flip
        if (fabs(current.q1 - current.q2) > 2 * M_PI && flip_time < 0.0f) {
            flip_time = t;
            break;
        }

        t += step_size;
    }

    flip_times[i] = flip_time;
}