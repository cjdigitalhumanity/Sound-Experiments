#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <string>
#include <complex>

using namespace std;

// Generate a vector of zeros
vector<double> zeros(int n) {
    return vector<double>(n, 0.0);
}

// Generate linearly spaced values
vector<double> linspace(double start, double stop, int num) {
    vector<double> result;

    if (num <= 0) return result;

    if (num == 1) {
        result.push_back(start);
        return result;
    }

    double step = (stop - start) / (num - 1);

    for (int i = 0; i < num; i++) {
        result.push_back(start + i * step);
    }

    return result;
}

// Check whether all Boolean values are true
bool all(const vector<bool>& x) {
    for (bool v : x) {
        if (!v) return false;
    }

    return true;
}

// Check whether vector elements are finite
vector<bool> checkFinite(const vector<double>& x) {
    vector<bool> result;

    for (double v : x) {
        result.push_back(std::isfinite(v));
    }

    return result;
}

// Calculate quadratic equation roots
vector<double> roots(double a, double b, double c) {
    vector<double> result;

    if (a == 0.0) {
        if (b != 0.0) {
            result.push_back(-c / b);
        }

        return result;
    }

    double discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return result;
    }

    double r1 = (-b + sqrt(discriminant)) / (2.0 * a);
    double r2 = (-b - sqrt(discriminant)) / (2.0 * a);

    result.push_back(r1);

    if (discriminant != 0.0) {
        result.push_back(r2);
    }

    return result;
}

const double PI = 3.14159265358979323846;

// EQ band specifications
struct BandSpec {
    double f0;
    double G_dB;
    double Q;
};

// Normalized filter coefficients
struct FilterCoefficients {
    double B0Norm;
    double B1Norm;
    double B2Norm;
    double A1Norm;
    double A2Norm;
};

int main() {

    // 1. Discrete-Time Specification Setup

    int sampleRate = 44100;  // Standard audio sample rate (must be > 2 * f0)
    double duration = 1.0;

    int num_samples =
        static_cast<int>(sampleRate * duration);

    // Equivalent to np.linspace(0, duration, num_samples, endpoint=False)
    vector<double> t(num_samples);

    for (int n = 0; n < num_samples; n++) {
        t[n] = static_cast<double>(n) / sampleRate;
    }

    vector<BandSpec> band_specs = {
        {1000,  6.0, 2.0},
        {3000, -4.0, 1.5}
    };

    // Frequency conversions

    double Nyquist = sampleRate / 2.0;

    // Create noisy composite wave (1 kHz tone + White Noise)

    vector<double> sine_wave(num_samples);

    for (int n = 0; n < num_samples; n++) {

        sine_wave[n] = std::sin(
            2.0 * PI * band_specs[0].f0 * t[n]
        );  // Matches f0 = 1000!
    }

    vector<double> noise(num_samples);

    random_device rd;
    mt19937 gen(rd());

    normal_distribution<double> distribution(0.0, 0.5);

    for (int n = 0; n < num_samples; n++) {
        noise[n] = distribution(gen);
    }  // Background hiss

    vector<double> composite_wave(num_samples);

    for (int n = 0; n < num_samples; n++) {
        composite_wave[n] = sine_wave[n] + noise[n];
    }

    // 2. Initialize EQ Bands

    vector<FilterCoefficients> eq_bands;

    for (const auto& spec : band_specs) {

        double f0 = spec.f0;
        double G_dB = spec.G_dB;
        double Q = spec.Q;

        // Input Parameter Validation

        if (!(0.0 < f0 && f0 < Nyquist)) {
            throw invalid_argument(
                "Invalid center frequency: " +
                to_string(f0) + " Hz. " +
                "Frequency must be between 0 and " +
                to_string(Nyquist) + " Hz."
            );
        }

        if (Q <= 0.0) {
            throw invalid_argument(
                "Invalid Q factor: " +
                to_string(Q) +
                ". Q must be greater than zero."
            );
        }

        double BW = f0 / Q;  // Bandwidth (Hz)

        double normalizedCenterFreq =
            (2.0 * PI * f0) / sampleRate;

        double angularFrequency =
            2.0 * PI * f0;

        // Linear Gain & Damping Factor

        double A = pow(10.0, G_dB / 40.0);

        double alpha =
            std::sin(normalizedCenterFreq) / (2.0 * Q);

        // Parametric Filter Design

        double cos_w = std::cos(normalizedCenterFreq);
        double sin_w = std::sin(normalizedCenterFreq);

        // Peaking EQ Number Coefficients

        double b_0 = 1.0 + (alpha * A);
        double b_1 = -2.0 * cos_w;
        double b_2 = 1.0 - (alpha * A);

        // Peaking EQ Denominator Coefficients

        double a_0 = 1.0 + (alpha / A);
        double a_1 = -2.0 * cos_w;
        double a_2 = 1.0 - (alpha / A);

        // Coefficient Normalization:

        double B0Norm = b_0 / a_0;
        double B1Norm = b_1 / a_0;
        double B2Norm = b_2 / a_0;

        double A1Norm = a_1 / a_0;
        double A2Norm = a_2 / a_0;
        double A0Norm = 1.0;

        // Filter Coefficient Validation

        vector<double> coefficients = {
            B0Norm, B1Norm, B2Norm,
            A1Norm, A2Norm
        };

        if (!all(checkFinite(coefficients))) {
            throw invalid_argument(
                "Invalid filter coefficients for center frequency " +
                to_string(f0) + " Hz."
            );
        }

        // Filter Stability Validation

        vector<double> poles = roots(
            1.0,
            A1Norm,
            A2Norm
        );

        bool unstable = false;

        for (double pole : poles) {
            if (std::fabs(pole) >= 1.0) {
                unstable = true;
                break;
            }
        }

        if (unstable) {
            throw invalid_argument(
                "Unstable filter detected at center frequency " +
                to_string(f0) + " Hz."
            );
        }

        eq_bands.push_back({
            B0Norm,
            B1Norm,
            B2Norm,
            A1Norm,
            A2Norm
        });
    }

    // 3. Cascaded Biquad Filter Processing

    vector<double> current_input = composite_wave;

    for (const auto& band : eq_bands) {

        // State History Buffer Initialization

        double x1 = 0.0;
        double x2 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;

        vector<double> y_band =
            zeros(static_cast<int>(current_input.size()));

        // biquad sample by sample sectioning

        for (int i = 0; i < static_cast<int>(current_input.size()); i++) {

            double x_n = current_input[i];

            // Calculate current output using band's normalized coefficients
            double y_n =
                (band.B0Norm * x_n) +
                (band.B1Norm * x1) +
                (band.B2Norm * x2) -
                (band.A1Norm * y1) -
                (band.A2Norm * y2);

            y_band[i] = y_n;

            // State History Updates
            x2 = x1;
            x1 = x_n;

            y2 = y1;
            y1 = y_n;
        }

        current_input = y_band;
    }

    vector<double> final_output = current_input;

    // 4. Calculate Frequency Response
    int num_points = 1000;

    vector<double> omega =
        linspace(0.0, PI, num_points);

    vector<complex<double>> e_neg_jw(num_points);
    vector<complex<double>> e_neg_j2w(num_points);

    const complex<double> j(0.0, 1.0);

    for (int i = 0; i < num_points; i++) {

        e_neg_jw[i] = exp(-j * omega[i]);

        e_neg_j2w[i] = exp(-2.0 * j * omega[i]);
    }

    vector<complex<double>> H_total_ejw(
        num_points,
        complex<double>(1.0, 0.0)
    );

    for (const auto& band : eq_bands) {

        for (int i = 0; i < num_points; i++) {

            complex<double> fdr_num =
                band.B0Norm +
                band.B1Norm * e_neg_jw[i] +
                band.B2Norm * e_neg_j2w[i];

            complex<double> fdr_den =
                1.0 +
                band.A1Norm * e_neg_jw[i] +
                band.A2Norm * e_neg_j2w[i];

            H_total_ejw[i] *= (fdr_num / fdr_den);
        }
    }

    // Magnitude response in decibels (dB)

    vector<double> magnitude_dB(num_points, 0.0);

    for (int i = 0; i < num_points; i++) {
        magnitude_dB[i] = 20.0 * log10(std::abs(H_total_ejw[i]));
    }
    return 0;
}
