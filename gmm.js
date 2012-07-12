
var Utils = (function (exports, Math) {
    // from http://stackoverflow.com/questions/55677/how-do-i-get-the-coordinates-of-a-mouse-click-on-a-canvas-element
    function relMouseCoords(event) {
        var totalOffsetX = 0;
        var totalOffsetY = 0;
        var canvasX = 0;
        var canvasY = 0;
        var currentElement = this;

        do {
            totalOffsetX += currentElement.offsetLeft;
            totalOffsetY += currentElement.offsetTop;
        } while(currentElement = currentElement.offsetParent)

        canvasX = event.pageX - totalOffsetX;
        canvasY = event.pageY - totalOffsetY;

        return {x:canvasX, y:canvasY}
    }

    HTMLCanvasElement.prototype.relMouseCoords = relMouseCoords;

    // f(N) is run a few times, timed and some stats are
    // returned as an object.
    function timeit(f, N) {
        var start, stop, dt;
        var worst = 0, best = 1000 * 3600, mean = 0, sigma = 0;
        var i, M = 7;
        for (i = 0; i < M; ++i) {
            start = Date.now();
            f(N);
            stop = Date.now();
            dt = stop - start;
            best = Math.min(best, dt);
            worst = Math.max(worst, dt);
            mean += dt;
            sigma += dt * dt;
        }

        mean /= M;
        sigma /= M;

        return { best: best, worst: worst, mean: mean, spread: Math.sqrt(sigma - mean * mean) };
    }

    exports.timeit = timeit;
    return exports;
})({}, Math);

var Transcribe = (function (exports, Math, Date) {

    try {
        Waveform.setup;
    } catch (e) {
        alert("gmm.js needs waveform.js included before it");
    }

    function elem(id) { return document.getElementById(id); }
    function animation_sequence(calls) {
        if (calls.length > 0) {
            calls[0]();
            calls.shift();
            webkitRequestAnimationFrame(function () { animation_sequence(calls); });
        }
    }

    function decimal_limiter(n) {
        return function (v) {
            var p = Math.pow(10, n);
            return Math.round(v * p) / p;
        };
    }

    function add_norms() {
        var N = arguments.length;
        var i, total = 0;
        for (i = 0; i < N; ++i) {
            total += arguments[i] * arguments[i];
        }
        return Math.sqrt(total);
    }

    var dec3 = decimal_limiter(3);
    var declim = [0, 1, 2, 3, 4, 5].map(decimal_limiter);
    var refFreq_Hz = 158.5, refFreqLow_Hz = refFreq_Hz / 4, refFreqHigh_Hz = refFreq_Hz *  16;
    var enable = true;
    var audio_context = new webkitAudioContext();
    var k_lpf_iterations = 2;

    // Calculates the periodicity of the signal at the
    // given frequency. 
    //
    // The returned result is a two-element array where the
    // second element gives the strength of the f component
    // and the first element gives the strength of the signal
    // itself. The first element might be useful for thresholding 
    // purposes to infer the utility of the first element.
    //
    // @param f32samples A Float32Array of samples (mono)
    // @param srate The sample rate of the audio signal in Hz.
    // @param window_secs The duration of the correlation window in seconds.
    // @param freq The frequency component to measure.
    // @param from_secs Starting time in seconds.
    function periodicity(f32samples, srate, window_secs, freq, from_secs) {
        // Get the index limits for the correlation operation.
        var ifrom   = Math.max(0, Math.min(Math.floor(from_secs * srate), f32samples.length - 1));
        var ito     = Math.max(ifrom + 1, Math.min(Math.floor((from_secs + window_secs) * srate), f32samples.length));

        if (ifrom >= ito) {
            return {
                freq:       0, 
                    next_freq:  0,
                    signal:     0,
                    strength:   0
            };
        }

        var i, s, c, p, a; // iteration variables

        var N = ito - ifrom;
        var P = 2; // Period ratio. Multiples of half periods give orthogonal sin/cos.

        var epsilon = 1.0 / 32768.0, epsilon2 = epsilon * epsilon; 
        ///< For preventing divide by zero.

        // Correct the frequency to fit an integral number of
        // cycles within the window. Only then can we assure the
        // orthogonality of the sin/cos components (see below).
        var periods = Math.round(P * freq * N / srate);
        freq = periods * srate / (P * N);

        var dp = freq * 2 * Math.PI / srate; 
        ///< Increment in phase for each time step.

        var sac = 0,  cac = 0;
        ///< Autocorrelation of sin and cos components with themselves.

        var sc = 0, cc = 0;
        ///< Correlation of sin and cos components with the signal.

        var ac = 0;
        ///< Autocorrelation of signal - i.e. correlation of signal with itself.

        var norm = 1.0 / (ito - ifrom);
        ///< Normalization factor for number of samples.

        // sin(2πft) and cos(2πft) form an orthogonal basis for the
        // f component of the signal. So we take dot product with both
        // and find the magnitude of the result vector to get the 
        // strength of the f component.
        for (i = ifrom; i < ito; ++i) {
            p = dp * (i - ifrom);
            a = f32samples[i];
            s = Math.sin(p);
            c = Math.cos(p);
            sc += s * a;
            cc += c * a;
            sac += s * s;
            cac += c * c;
            ac += a * a;
        }

        var x = cc / Math.sqrt(epsilon2 + cac * ac);
        var y = sc / Math.sqrt(epsilon2 + sac * ac);
        var energy = x * x + y * y;

        return { 
            freq:       freq, 
            next_freq:  ((periods + 1) * srate / (P * N)), 
            signal:     Math.sqrt(ac * norm), 
            strength:   Math.sqrt(energy),
            energy:     energy,
            relative_energy: norm * (sc * sc + cc * cc) / (epsilon2 + ac),
            phase:      Math.atan2(y, x)
        };
    }

    // at 0, = 0
    // at 1, = 1
    // at 0.5 = 0.5
    function raised_cosine(x) {
        return 0.5 * (1 + Math.cos((x - 1) * Math.PI));
    }

    // measure is a function (f,t) -> { freq:, next_freq:, signal:, strength: }
    // To get such a function, you can wrap the above "periodicity" function in a closure.
    function spectrum(measure, time_secs, fromFreq_Hz, toFreq_Hz) {
        var f = fromFreq_Hz;
        var result = {components: [], strength: -10.0, signal: -10.0};
        var components = result.components, component;
        for (f = fromFreq_Hz; f < toFreq_Hz;) {
            component = measure(f, time_secs);
            components.push(component);
            f = component.next_freq;
            result.strength = Math.max(result.strength, component.strength);
            result.signal = Math.max(result.signal, component.signal);
        }

        // Window the spectrum towards the edges so we don't get
        // runaway gaussians.
        if (components.length > 8) {
            var i, N, w;
            for (i = 0, N = components.length - 1; i < 4; ++i) {
                w = raised_cosine(i/4);
                components[i].energy *= w;
                components[N - i].energy *= w;
            }
        }

        return result;
    }

    function find_freq_index(spec, f) {
        var i, N, dist, minDist = 44100.0, ix;
        for (i = 0, N = spec.length; i < N; ++i) {
            dist = Math.abs(f - spec[i].freq);
            if (dist < minDist) { minDist = dist; ix = i; }
        }
        return ix;
    }

    function find_freq(spec, f) {
        var i = find_freq_index(spec, f);
        return (i && spec[i]) || {strength: 0.0};
    }

    // Gives gaussian model of energy peaks in interval given by
    // freq_Hz * pow(2, leftWindow_st/12) to freq_Hz * pow(2, rightWindow_st/12)
    //
    // The mesure(f,t) function is expected to measure the frequency component
    // near the time t.
    //
    // time_secs is the time at which this peak estimation is done.
    function find_peaks_around(freq_Hz, leftWindow_st, rightWindow_st, measure, time_secs) {
        var fromFreq_Hz = freq_Hz * Math.pow(2, leftWindow_st / 12);
        var toFreq_Hz = freq_Hz * Math.pow(2, rightWindow_st / 12);
        var result = find_peaks_around_hz(fromFreq_Hz, toFreq_Hz, measure, time_secs);
        result.window = [leftWindow_st, rightWindow_st];
        return result;
    }

    function find_peaks_around_hz(fromFreq_Hz, toFreq_Hz, measure, time_secs) {
        var result = find_initial_peaks_around_hz(fromFreq_Hz, toFreq_Hz, measure, time_secs);
        result = GMM.stabilize(result.spectrum, result);
        //stabilize_and_simplify_gaussians(result.comps, result.orig_peaks);
        //delete result.comps;
        return result;
    }

    // Runs a 121 filter on the energy part of comps N times
    // and returns the result as a Float32Array of the same length
    // as comps. If array is given, it uses that instead of creating
    // a new one. Note that array.length must be the same as comps.length.
    //
    // spectrum = {power:, phase:}
    function low_pass_filter_spectral_energy(spectrum, N) {
        var i, j, len, v1, v2, v3;

        // Keep the original phase around.
        var result = mk_spectrum(spectrum.energy.length, null, spectrum.phase);

        // Copy the energy. The caller can be assured that in all cases
        // they get a different buffer and need not be afraid that
        // the original buffer may get modified. 
        //
        // Of course, the phase is passed without copying.
        var energy = result.energy;
        energy.set(spectrum.energy, 0);

        if (N === 0) {
            return result;
        }


        for (j = 0; j < N; ++j) {
            v1 = 0;
            v2 = energy[0];
            v3 = 0;

            for (i = 0, len = energy.length - 1; i < len; ++i) {
                v3 = energy[i + 1];
                energy[i] = 0.25 * v1 + 0.5 * v2 + 0.25 * v3;
                v1 = v2;
                v2 = v3;
            }

            energy[energy.length - 1] = 0.25 * v1 + 0.5 * v2;
        }

        return result;
    }

    // weight = weight of the centre sample.
    // Weight of surrounding samples is derived from
    // whatever is left over.
    function low_pass_filter_spectrum(comps, weight) {
        var w1 = 0.5 * (1 - weight), w2 = weight, w3 = w1;
        var v1 = 0, v2 = comps[0].energy, v3 = 0;
        var i, len;
        for (i = 0, len = comps.length - 1; i < len; ++i) {
            v3 = comps[i + 1].energy;
            comps[i].energy = v1 * w1 + v2 * w2 + v3 * w3;
            v1 = v2;
            v2 = v3;
        }
        v3 = 0;
        comps[comps.length - 1].energy = w1 * v1 + w2 * v2 + w3 * v3;
        return comps;
    }

    // new Spectrum(N)
    // new Spectrum(fromFreq_Hz, toFreq_Hz, measure, time_secs)
    function Spectrum() {
        var N;
        switch (arguments.length) {
            case 0: 
                this.energy = this.phase = null;
                return this;
            case 1: 
                N = arguments[0];
                this.energy = new Float32Array(N);
                this.phase = new Float32Array(N);
                this.reassigned_energy = new Float32Array(N);
                this.__reassigned_frequency = new Float32Array(N);
                this.length = N;
                return this;
            case 4:
                return this.measure(arguments[0], arguments[1], arguments[2], arguments[3]);
            default:
                throw "Spectrum: Invalid argument pattern";
        }
    }

    // Instantiate like this -
    //
    // var spectrum = (new Spectrum()).setup(sampleRate, windowLen, fromFreq_Hz, toFreq_Hz)
    // From then onwards, you can call -
    //      spectrum.update(t, samples) 
    // to calculate the spectrum for the given samples over a window of the previously
    // specified windowLen, starting from time t (given in seconds).
    //
    Spectrum.prototype.setup = function (sampleRate, windowLen, fromFreq_Hz, toFreq_Hz) {
        var fromPeriods = Math.floor(2 * fromFreq_Hz * windowLen / sampleRate);
        var toPeriods = Math.floor(2 * toFreq_Hz * windowLen / sampleRate) + 1;
        var N = toPeriods - fromPeriods;

        this.sample_rate = sampleRate;
        this.window_length = windowLen;

        this.energy = new Float32Array(N);
        this.phase = new Float32Array(N);
        this.reassigned_energy = new Float32Array(N);
        this.__reassigned_frequency = new Float32Array(N);
        this.length = N;
        this.fromFreq_Hz = sampleRate * fromPeriods / (2 * windowLen);
        this.toFreq_Hz = sampleRate * toPeriods / (2 * windowLen);
        this.df = sampleRate / (2 * windowLen);
        this.frequency = function (i) { return sampleRate * (fromPeriods + i) / (2 * windowLen); };
        this.reassigned_frequency = function (i) { return this.__reassigned_frequency[i]; };

        var Wext = Math.ceil(windowLen * 1.5);
        var sincos_tables = {sin: [], cos: []};
        var i, j, f, st, ct, n;
        for (i = fromPeriods; i < toPeriods; ++i) {
            st = new Float32Array(windowLen);
            ct = new Float32Array(windowLen);

            f = Math.PI * i / windowLen;
            n = 1 / Math.sqrt(0.5 * windowLen);

            for (j = 0; j < windowLen; ++j) {
                st[j] = - n * Math.sin(f * j);
                ct[j] = n * Math.cos(f * j);
            }

            sincos_tables.sin.push(st);
            sincos_tables.cos.push(ct);
        }

        // Initially no reassignment done.
        for (i = 0; i < N; ++i) {
            this.__reassigned_frequency[i] = this.frequency(f);
        }

        // Prepare temp tables for spectrum calculation.
        var sin_acc = new Float32Array(Wext + 1);
        var cos_acc = new Float32Array(Wext + 1);
        var power_acc = new Float32Array(Wext + 1);
        var kSmallAngle = Math.PI / 5;

        // The main function that calculates the spectrum for the interval [t,t+W].
        function update(t, samples) {

            // If samples is not given, check whether it is available as part of object state.
            samples = samples || this.samples;
            if (!samples) {
                throw "Spectrum.update: No samples array specified.";
            }

            var f, f2, i, j, ssum, csum, ssum2, csum2, power, offset, phase, dphase, x, y, y2, s;
            var e = this.energy;
            var p = this.phase;
            var rf = this.__reassigned_frequency;
            var W = windowLen - windowLen % 1;
            var extWindowLen = W + Math.ceil(W / 2);
            var slideWindowLen = Math.floor(W / 2);
            var start = Math.min(Math.max(0, Math.floor(t * sampleRate)), samples.length - extWindowLen);

            if (start < 0) {
                throw "Too short a signal for measurement";
            }

            this.time_secs = start / sampleRate;

            var st, ct;

            // Zero the reassigned energy
            for (f = 0; f < N; ++f) {
                this.reassigned_energy[f] = 0;
            }

            // Accumulate the power first so we don't have to
            // do this for every frequency.
            for (j = 0, power = 0; j < extWindowLen; ++j) {
                power_acc[j] = power;
                power += samples[start + j] * samples[start + j];
            }
            power_acc[j] = power;

            // For each frequency ...
            for (f = 0; f < N; ++f) {
                st = sincos_tables.sin[f];
                ct = sincos_tables.cos[f];

                // Calculate the integral of the power and sin and cos products.
                for (j = 0, ssum = 0, csum = 0; j < extWindowLen; ++j) {
                    sin_acc[j] = ssum;
                    cos_acc[j] = csum;

                    s = samples[start + j];
                    ssum += st[j % W] * s;
                    csum += ct[j % W] * s;
                }

                sin_acc[j] = ssum;
                cos_acc[j] = csum;

                // Determine energy and phase from the first W components
                // of the integral.
                ssum = sin_acc[W] - sin_acc[0];
                csum = cos_acc[W] - sin_acc[0];
                e[f] = (ssum * ssum + csum * csum) / (power_acc[W] - power_acc[0]);
                p[f] = Math.atan2(ssum, csum);

                // One more extra filter.
                if (true) {
                    for (j = 0; j < slideWindowLen; ++j) {
                        sin_acc[j] = sin_acc[j + W] - sin_acc[j];
                        cos_acc[j] = cos_acc[j + W] - cos_acc[j];
                    }

                    for (j = 0, ssum = 0, csum = 0; j < extWindowLen; ++j) {
                        ssum2 = sin_acc[j];
                        csum2 = cos_acc[j];
                        sin_acc[j] = ssum;
                        cos_acc[j] = csum;
                        ssum += ssum2;
                        csum += csum2;
                    }

                    sin_acc[j] = ssum;
                    cos_acc[j] = csum;
                }

                // Slide by one sample and measure phase difference upon sliding.
                // Accumulate these phase differences over a period of W/2 and
                // average to determine the average rate at which the phase changes
                // around this time. That gives us the "frequency offset" from f-th
                // frequency, which we store in the reassignment table.
                for (offset = 1, phase = 0, ssum = sin_acc[W] - sin_acc[0], csum = cos_acc[W] - cos_acc[0]; offset < slideWindowLen; ++offset) {
                    ssum2 = sin_acc[offset + W] - sin_acc[offset];
                    csum2 = cos_acc[offset + W] - cos_acc[offset];

                    // complex2 x complex1* gives a complex number whose phase
                    // is the phase offset from complex1 to complex2.
                    x = csum * csum2 + ssum * ssum2;
                    y = csum * ssum2 - ssum * csum2;
                    if (y/x > kSmallAngle || y/x < -kSmallAngle) {
                        phase += Math.atan2(y,x);
                    } else {
                        y /= x;
                        y2 = y * y;
                        phase += y * (1 - y2 * (0.33333 - 0.2 * y2)); // An approximation for Math.atan2 for small angles.
                    }

                    csum = csum2;
                    ssum = ssum2;
                }

                // Average rate of change of phase.
                dphase = phase * sampleRate / (2 * Math.PI * slideWindowLen);
                rf[f] = this.frequency(f) + dphase;

                // Calculate the reassigned energy.
                f2 = (rf[f] - this.fromFreq_Hz) / this.df;
                if (f2 >= 0 && f2 < N - 1) {
                    this.reassigned_energy[f2 - f2 % 1] += e[f] * (1 - f2 % 1);
                    this.reassigned_energy[f2 - f2 % 1 + 1] += e[f] * (f2 % 1);
                }
            }

            // Taper towards the ends so we don't end up guessing around those regions too much.
            for (f = 0, n = Math.min(4, N / 2); f < n; ++f) {
                s = raised_cosine(f/4);
                this.energy[f] *= s;
                this.reassigned_energy[f] *= s;
                this.energy[N - f - 1] *= s;
                this.reassigned_energy[N - f - 1] *= s;
            }

            this.power = (power_acc[W] - power_acc[0]) / W;
            return this;
        }

        this.update = update;
        return this;
    };

    Spectrum.prototype.clone = function () {
        if (this.energy) {
            var s = new Spectrum(this.energy.length);
            s.energy.set(this.energy, 0);
            s.phase.set(this.phase, 0);
            s.reassigned_energy.set(this.reassigned_energy, 0);

            s.power          = this.power; // The total signal power (square of strength).
            s.fromFreq_Hz    = this.fromFreq_Hz; // Actual starting frequency.
            s.toFreq_Hz      = this.toFreq_Hz; // Actual ending frequency
            s.df             = this.df; // The frequency delta between two consecutinve samples.
            s.length         = this.length; // The number of samples.
            s.frequency      = this.frequency;
            s.reassigned_frequency = this.reassigned_frequency;
            s.__reassigned_frequency.set(this.__reassigned_frequency, 0);
            s.update         = this.update;

            return s;
        } else {
            throw "Clone of unintialized spectrum!";
        }
    };

    // Not an FFT, but could be one. For the moment,
    // it is band limited. See the last structure
    // in the return statement for what this returns.
    Spectrum.prototype.measure = function (fromFreq_Hz, toFreq_Hz, measure, time_secs) {
        this.fromFreq_Hz = fromFreq_Hz;
        this.toFreq_Hz = toFreq_Hz;
        this.time_secs = time_secs;

        var e = 0, rele = 0, i, j, c, f, f1, f2, df, N;

        c = measure(fromFreq_Hz, time_secs);
        f1 = c.freq;
        df = c.next_freq - f1;
        N = Math.ceil((toFreq_Hz - f1) / df);

        var energy = new Float32Array(N);
        var reassigned_energy = new Float32Array(N);
        var phase = new Float32Array(N);
        var reassigned_freq = new Float32Array(N);
        this.__reassigned_frequency = reassigned_freq;
        this.energy = energy;
        this.phase = phase;
        this.reassigned_energy = reassigned_energy;

        for (i = 0, f2 = f1; f2 < toFreq_Hz; ++i) {
            c = measure(f2, time_secs);
            energy[i] = c.energy;
            phase[i] = c.phase;
            reassigned_freq[i] = c.reassigned_freq;
            e += c.energy;
            rele += c.relative_energy;
            f2 = c.next_freq;
        }

        function calculate_reassigned_energy() {
            var i, j, f, frac;
            for (i = 0; i < reassigned_energy.length; ++i) {
                reassigned_energy[i] = 0;
            }

            for (i = 0; i < energy.length; ++i) {
                f = (reassigned_freq[i] - f1) / df;
                if (f >= 0 && f + 1 < energy.length) {
                    frac = f % 1;
                    j = f - frac;
                    reassigned_energy[j] += energy[i] * (1 - frac);
                    reassigned_energy[j + 1] += energy[i] * frac;
                }
            }

            return reassigned_energy;
        }

        this.power          = e; // The total signal power (square of strength).
        this.relative_power = rele; // The power in the band of measurement as a fraction of the total power.
        this.fromFreq_Hz    = f1; // Actual starting frequency.
        this.toFreq_Hz      = f2; // Actual ending frequency
        this.df             = df; // The frequency delta between two consecutinve samples.
        this.length         = i; // The number of samples.
        this.frequency      = function (i) { return f1 + i * df; };
        this.reassigned_frequency = function (i) { return reassigned_freq[i]; };

        // A method that will use the same setup and measure for a different time.
        this.update = function (t) {
            var i, e, rele, c, N;

            for (i = 0, e = 0, rele = 0, N = this.length; i < N; ++i) {
                c = measure(f1 + i * df, t);
                energy[i] = c.energy;
                phase[i] = c.phase;
                reassigned_freq[i] = c.reassigned_freq;
                e += c.energy;
                rele += c.relative_energy;
            }

            this.power = e;
            this.relative_power = rele;
            this.time_secs = t;

            calculate_reassigned_energy();
            return this;
        };

        calculate_reassigned_energy();
        return this;
    }

    // Creates a filter that can just be multiplied on to the spectrum.
    Spectrum.prototype.make_comb_filter = function (fundamental_Hz, NHarmonics) {
        if (!this.length) {
            throw "make_comb_filter: Uninitialized spectrum!";
        }

        var s = new Spectrum(this.length);

        var fc, spread, i, j, M, N;

        N = s.length;

        for (i = 0, M = NHarmonics; i < M; ++i) {
            fc = fundamental_Hz * (i + 1);
            spread = fc * (Math.pow(2, 1/6) - 1);
            for (j = 0; j < N; ++j) {
                s.energy[i] += gaussian(this.frequency(j), 1, fc, spread);
            }
        }

        return s;
    };

    // Takes an array of frequencies and makes one comb filter for
    // each of those frequencies and its N harmonics.
    Spectrum.prototype.make_comb_filters = function (freqs, NHarmonics) {
        var spec = this;
        return freqs.map(function (f) {
            return spec.make_comb_filter(f, NHarmonics);
        });
    };

    // Low pass filter of energy - 121 type, applied N times.
    Spectrum.prototype.lpf = function (N) {
        var i, j, len;
        var v1, v2, v3, rv1, rv2, rv3;
        for (j = 0; j < N; ++j) {
            v1 = rv1 = 0;
            v2 = this.energy[0];
            rv2 = this.reassigned_energy[0];
            v3 = rv3 = 0;
            for (i = 0, len = this.length - 1; i < len; ++i) {
                v3 = this.energy[i + 1];
                rv3 = this.reassigned_energy[i + 1];
                this.energy[i] = 0.25 * v1 + 0.5 * v2 + 0.25 * v3;
                this.reassigned_energy[i] = 0.25 * rv1 + 0.5 * rv2 + 0.25 * rv3;
                v1 = v2;
                v2 = v3;
                rv1 = rv2;
                rv2 = rv3;
            }

            this.energy[this.length - 1] = 0.25 * v1 + 0.5 * v2;
            this.reassigned_energy[this.length - 1] = 0.25 * rv1 + 0.5 * rv2;
        }

        return this;
    };

    // Takes a filter f of type Spectrum and applies its
    // 'energy' part to this spectrum's energy.
    Spectrum.prototype.filter = function (f) {
        var e = this.energy;
        var fe = f.energy;

        var N = this.length;
        if (N !== f.length) {
            throw "Incompatible filter";
        }

        var i;
        for (i = 0; i < N; ++i) {
            e[i] *= fe[i];
        }

        return this;
    };

    // Blind decimation of given samples of Float32.
    function decimate(samples, step) {
        var N = Math.floor(samples.length / step);
        var i;

        var result = new Float32Array(N);
        for (i = 0; i < N; ++i) {
            result[i] = samples[i * step];
        }

        return result;
    }

    // In-place box filter the given array of samples.
    function box_filter(samples, windowLen) {
        var i, j, len, acc, a;
        var delay_line = new Float32Array(windowLen);
        var norm = 1.0 / windowLen;

        for (i = 0, j = 0, len = samples.length, acc = 0; i < len; ++i, j = (j + 1) % windowLen) {
            a = samples[i];
            acc += a - delay_line[j];
            delay_line[j] = a;
            samples[i] = acc * norm;
        }

        delete delay_line;
        return samples;
    }

    // Gives a weighting factor to use which will weight larger
    // values of the given octave number lower. The peak will
    // be 1, when octave = 0 and will diminish to around 0.5
    // when octave is around 50 cents (= 50/1200 ~ 0.04). The
    // idea is to use this as an additional weight to determine
    // whether a measured drequency component is a harmonic 
    // or not.
    function harmonic_bias(octave) {
        octave /= (50 / 1200); // a standard deviation of 50 cents.
        return Math.exp(- 0.5 * octave * octave);
    }

    // Runs an N-harmonic heterodyne filter of the signal and does a 
    // sliding window average on the result, accumulating the resulting
    // signal in the member array "baseband".
    function heterodyne(buffer, from_secs, to_secs, window_secs, step_frac, fundamental_Hz, NHarmonics) {
        var samples = buffer.getChannelData(0);

        var windowLen = Math.max(1, Math.round(window_secs * buffer.sampleRate));
        var step = windowLen; // FORCE TO windowLen FOR NOW! // Math.max(1, Math.round(step_frac * window_secs * buffer.sampleRate));

        var N = samples.length;
        var fromIndex = Math.max(0, Math.min(N - windowLen, Math.round(from_secs * buffer.sampleRate)));
        var toIndex = Math.min(N, Math.max(fromIndex, Math.ceil(to_secs * buffer.sampleRate)));
        var range = toIndex - fromIndex;

        // First heterodyne the signal down to baseband and
        // then do the running window filter on it.
        var i, count;

        // Prepare the sine and cos waves so we can compute the
        // heterodyning efficiently.
        var period      = Math.round(buffer.sampleRate / fundamental_Hz);
        var adjFundamental_Hz = buffer.sampleRate / period;
        var dphase      = 2 * Math.PI / period;
        var dt          = 1.0 / buffer.sampleRate;

        // Run the window average.
        var kErrorAngle      = 2 * Math.PI * windowLen;
        var windowNorm      = 1 / windowLen;
        var invWindowNorm   = windowLen;
        var phaseNorm       = 1 / (2 * Math.PI * dt);
        var h, a, a1, a2, s, c, ds, dc, c1, c2, s1, s2, x, y, r, angle, dphaseh, phaseNormh, dphaseArr, rArr, rrelArr;
        var cosAcc, sinAcc, cos2Acc, sin2Acc, sincosAcc, eAcc;
        var dphaseAvg = [], rAvg = [], freqs = [], periods = [], sinarr = [], cosarr = [];
        var sinarrh, cosarrh, periodh, freqh;
        dphaseAvg.length = NHarmonics;
        for (h = 0; h < NHarmonics; ++h) {
            dphaseAvg[h] = new Float64Array(range);
            rAvg[h] = new Float64Array(range);
            periods[h] = Math.round(buffer.sampleRate / ((h + 1) * fundamental_Hz));
            freqs[h] = buffer.sampleRate / periods[h];
            sinarr[h] = new Float32Array(periods[h]);
            cosarr[h] = new Float32Array(periods[h]);
            for (i = 0, count = periods[h]; i < count; ++i) {
                sinarr[h][i] = Math.sin(2 * Math.PI * i / count);
                cosarr[h][i] = Math.cos(2 * Math.PI * i / count);
            }
        }

        var kSmallAngle = Math.PI / 5;
        var ih, iwh, ß;

        // Compute windowed energy for reference.
        rrelArr = new Float32Array(range);
        for (i = 0; i < range; ++i) {
            rrelArr[i] = samples[fromIndex + i] * samples[fromIndex + i];
        }
        box_filter(rrelArr, windowLen);
        rrelArr = decimate(rrelArr, step);

        for (h = 0; h < NHarmonics; ++h) {
            cosAcc = sinAcc = cos2Acc = sin2Acc = sincosAcc = eAcc = 0;
            dphaseh = (h + 1) * dphase;
            sinarrh = sinarr[h];
            cosarrh = cosarr[h];
            periodh = periods[h];
            freqh = freqs[h];
            //phaseNormh = phaseNorm / (h + 1);

            for (i = 0; i < windowLen; ++i) {
                a = samples[fromIndex + i];
                c = a * cosarrh[i % periodh];
                s = - a * sinarrh[i % periodh];
                eAcc += a * a;
                cosAcc += c;
                sinAcc += s;
                //cos2Acc += c * c;
                //sin2Acc += s * s;
                //sincosAcc += c * s;
            }

            dphaseArr = dphaseAvg[h];
            rArr = rAvg[h];

            for (i = 1; i < range; ++i) {
                a1 = samples[fromIndex + i - 1];
                a2 = samples[fromIndex + i + windowLen - 1];
                ih = (i - 1) % periodh;
                iwh = (i + windowLen - 1) % periodh;
                c1 = cosarrh[ih] * a1;
                c2 = cosarrh[iwh] * a2;
                dc = c2 - c1;
                s1 = - a1 * sinarrh[ih];
                s2 = - a2 * sinarrh[iwh];
                ds = s2 - s1;

                // dphase = angle of {(cosAvg[i] + j sinAvg[i]) x (cosAvg[i-1] - j sinAvg[i-1])}
                r = cosAcc * cosAcc + sinAcc * sinAcc;
                x = r + dc * cosAcc + ds * sinAcc;
                y = cosAcc * ds - sinAcc * dc;

                eAcc += a2 * a2 - a1 * a1;
                cosAcc += dc;
                sinAcc += ds;
                //cos2Acc += c2 * c2 - c1 * c1;
                //sin2Acc += s2 * s2 - s1 * s1;
                //sincosAcc += s2 * c2 - s1 * c1;

                // For small changes, we expect angle to be approximately
                // = y/r. If angle differs from y/r by more than 10%, we
                // flag that as a discontinuity since this difference is only
                // over a time step of a single sample. The angle of 36 degrees
                // is around the point where tan(x) differs from x by more around
                // 10%, so we use PI/5 as the threshold.
                if (y / x > kSmallAngle || y / x < -kSmallAngle) {
                    // We're only stepping by one sample. If the
                    // angle changes by more than 36 degrees, 
                    // it implies some kind of discontintuity or
                    // a region of such low energy as to be useless.
                    // In such cases, we flag by setting the angle to
                    // 2πn so that the calculated frequency will be
                    // > the sampling rate, which we can use to
                    // detect such regions later on since the sampling
                    // rate is an absurd value for an audio frequency.
                    angle = Math.atan2(y, x);
                } else {
                    // We're guaranteed here that the angle is "small enough"
                    // according to the definition above. In this case, 
                    // atan2 can be approximated to around 1% error by 
                    // the formula -
                    //
                    // angle = (y/x) (1 - (1/3)ß + (1/5)ß^2)
                    // where ß = (y/x)^2
                    angle = y/x;
                    ß = angle * angle;
                    angle *= (1 - 0.333333 * ß + 0.2 * ß * ß);
                }

                // dphaseAvg directly gives the frequency difference in Hz
                // since phaseNorm includes the 2π factor and the sampling rate.
                rArr[i] = r * windowNorm * windowNorm;
                dphaseArr[i] = angle * phaseNorm;
            }
        }

        // Run a box filter on dphaseAvg to get rid of fluctuations.
        // Note: Wondering whether this is necessary, but it seems
        // to be necessary to add the phase differences and estimate
        // frequency at the given window stepping rate. Also note that
        // the "exceptional value" of 2πn set in the loop above will
        // cause the result of the box filtered value to be near the
        // sampling rate and therefore will still be an absurd value
        // > the sampling rate, since step < windowLen. Note that the
        // effect of such an exceptional value will last for "windowLen"
        // samples and therefore if you simply decimate the result,
        // you could end up with some consecutive such absurdities
        // most of the time. 
        //
        // WARNING: If you have too many of those occur too frequently, 
        // then the whole thing can be a disaster. So need to watch out.
        //
        for (h = 0; h < NHarmonics; ++h) {
            dphaseAvg[h] = decimate(box_filter(dphaseAvg[h], windowLen), step);
            rAvg[h] = decimate(rAvg[h], step); // decimate(box_filter(rAvg[h], windowLen), step);
        }

        return {
            dphase:     dphaseAvg,
            r:          rAvg,
            length:     dphaseAvg[0].length,
            freq:       function (h, i) {
                return freqs[h] + dphaseAvg[h][i];
            },
            pitch_est:  function (i, harmonic_weight) {
                var w = 0, wf = 0, wff = 0, f0, fn, f, r, he, hw, df;
                var h, len, hn;
                harmonic_weight = (arguments.length < 2 ? 1 : harmonic_weight);
                f0 = freqs[0] + dphaseAvg[0][i];
                df = 0.25 * buffer.sampleRate / windowLen;
                for (h = 0, hn = 0, he = 0; h < NHarmonics; ++h) {
                    f = dphaseAvg[h][i];            // Estimate of frequency difference.
                    r = rAvg[h][i];                 // Estiamte of harmonic energy for this component.
                    if (2 * f < buffer.sampleRate) {
                        ++hn;
                        he += r;
                        hw = h * harmonic_weight + 1;
                        fn = freqs[h] + f;
                        f = fn / f0;                    // Relative to fundamental.
                        f = f / Math.round(f);          // Down to base harmonic.
                        f = Math.log(f) / Math.LN2;  // To octaves.
                        hw *= harmonic_bias(f);
                        w += r * hw;
                        wf += r * hw * f;
                        wff += r * hw * f * f;
                    }
                }

                // If harmonicity is zero, there is no point
                // trying to estimate a pitch.
                if (hn > 0) {
                    w = Math.max(1e-8, w);
                    wf /= w;
                    wff /= w;

                    return {
                        mean_hz: f0 * Math.pow(2, wf),
                            mean_cents: Math.round(1200 * (Math.log(f0 / freqs[0]) / Math.LN2 + wf)),
                            herr_cents: Math.round(1200 * Math.sqrt(wff - wf * wf)), 
                            // harmonic error +/- around the mean. 
                            // Usually multiplied by 3 for the 3σ rule.
                            harmonicity: hn, // Number of harmonics available and used for the estimate.
                            energy_dbm: Math.round(10000 * 0.5 * Math.log(he)/Math.log(10)), 
                            // The harmonic energy in milli-dB.
                            rel_energy_dbm: Math.round(10000 * 0.5 * (Math.log(he) - Math.log(rrelArr[i])) / Math.log(10)),
                            // Save away the whole object in the 'details' field for future reference.
                            details: this
                    };
                } else {
                    return null;
                }
            },
            fundamental_hz:    freqs[0],
            freqs:          freqs,
            periods:        periods,
            window_length:  windowLen,
            step:       step, 
            time:       function (i) { return (fromIndex + i * step) / buffer.sampleRate; },

            sample_rate: buffer.sampleRate,

            src: {
                buffer: buffer,
                from_secs: from_secs,
                from_index: fromIndex,
                to_secs: to_secs,
                to_index: toIndex,
                window_secs: window_secs,
                step_frac: step_frac
            }
        };
    }

    // Same as heterodyne, except that fundamental_hz is an array of fundamental frequencies
    // and the result is an array of objects corresponding to those fundamental frequencies.
    function multi_heterodyne(buffer, from_secs, to_secs, window_secs, step_frac, fundamental_Hz, NHarmonics) {
        var h = fundamental_Hz.map(function (f) { 
            return heterodyne(buffer, from_secs, to_secs, window_secs, step_frac, f, NHarmonics);
        });

        h.pitch_est = function (i, harmonic_weight, min_harmonicity) {
            var p = h.map(function (r) { return r.pitch_est(i, harmonic_weight); });
            var pmax = null;
            var k, pk;
            for (k = 0; k < h.length; ++k) {
                pk = p[k];
                if (pk) {
                    if (pk.harmonicity > min_harmonicity) {
                        if (pmax) {
                            if (pmax.energy_dbm < pk.energy_dbm) {
                                pmax = pk;
                            }
                        } else {
                            pmax = pk;
                        }
                    }
                }
            }

            return pmax;
        };

        return h;
    }

    // Returns an array of frequencies given the root in Hz and values in semitones.
    // There are two argument pattern overloads - 
    //
    // frequencies(root, start_semitone, end_semitone, step_semitone)
    //      and
    // frequencies(root, [array-of-semitones])
    function frequencies(root) {
        if (arguments.length === 4) {
            // frequencies(root, start_semitone, end_semitone, step_semitone)
            var start_semitone = arguments[1], end_semitone = arguments[2], step_semitone = arguments[3];
            var fs = [];
            var f;
            for (f = start_semitone; f <= end_semitone; f += step_semitone) {
                fs.push(root * Math.pow(2, f/12));
            }
            return fs;
        } else if (arguments.length === 2) {
            // frequencies(root, [array-of-semitones])
            var semitones = arguments[1];
            return semitones.map(function (s) { return root * Math.pow(2, s/12); });
        }
    }

    // A numerically approximated gaussian integral function.
    // It is a function (x) where x is expressed with unity sigma
    // and zero mean. So if you want the integral for a distribution
    // with mean m and variance s, you call gaussian_integral((x - m) / s)
    //
    // The function gives the probability of a random variable x
    // falling anywhere in the range (-inf, x]. Therefore if you
    // want to find out the probability of the random variable falling
    // in the range x1 to x2, you fo F(x2) - F(x1).
    var gaussian_integral = (function () {
        // We calculate up to 4 sigma.
        var Ns = 64, Nmid = Ns * 4, N = Ns * 8, norm;
        var samples = new Float32Array(N);
        var i, x;

        // Initialize the array with gaussian values.
        for (i = 0, norm = Math.log(1 / Math.sqrt(2 * Math.PI)); i < N; ++i) {
            x = (i - Nmid) / Ns;
            samples[i] = Math.exp(norm - 0.5 * x * x);
        }

        // Integrate the array.
        for (i = 1; i < N; ++i) {
            samples[i] += samples[i - 1];
        }

        // Normalize the whole thing so that the last value is 1.
        for (i = 0, norm = 1 / samples[N - 1]; i < N; ++i) {
            samples[i] *= norm;
        }

        // Construct the function that abstracts away the
        // discrete implementation underneath.
        return function (x) {
            var i = x * Ns + Nmid;
            var fi = Math.floor(i);

            if (fi < 0) {
                return 0;
            }

            if (fi >= N - 1) {
                return 1;
            }

            var w = i - fi;


            // Use linear interpolation between values.
            return samples[fi + 1] * w + samples[fi] * (1 - w);
        };
    })();


    // Gaussian mixture model of N gaussians
    function GaussianMixtureModel() {
        var N = 0, Nmax = 16;
        var me = this;
        var weight      = new Float64Array(Nmax);
        var mean        = new Float64Array(Nmax);
        var precision   = new Float64Array(Nmax); // = 1 / sigma^2
        var norm        = new Float64Array(Nmax);

        function realloc(new_nmax) {
            var w, m, p, n;

            Nmax = new_nmax;

            // Realloc.
            w = new Float64Array(Nmax);
            m = new Float64Array(Nmax);
            p = new Float64Array(Nmax);
            n = new Float64Array(Nmax);

            w.set(weight, 0);
            m.set(mean, 0);
            p.set(precision, 0);
            n.set(norm, 0);

            weight = w;
            mean = m;
            precision = p;
            norm = n;

            return me;
        }

        // Add a new gaussian. Weight, mean and precision given.
        function add(w, µ, Ω) {
            var w2, m2, p2, n2;
            if (N < Nmax) {
                weight[N] = w;
                mean[N] = µ;
                precision[N] = Ω;
                norm[N] = Math.sqrt(Ω / (2 * Math.PI));
                ++N;
                me.length = N;
                return me;
            } else {
                return realloc(Nmax * 2).add(w, µ, Ω);
            }
        };

        // Remove the i-th gaussian from the model.
        function remove(i) {
            if (i >= N) { return this; }

            var j;
            for (j = i + 1; j < N; ++j) {
                weight[j-1]     = weight[j];
                mean[j-1]       = mean[j];
                precision[j-1]  = precision[j];
                norm[j-1]       = norm[j];
            }

            --N;
            me.length = N;
            return this;
        }

        // Same as value_of but lets you override the precision.
        function prec_value_of(i, x, prec) {
            x = (x - mean[i]);
            x *= x * prec;
            return weight[i] * Math.sqrt(prec/(2*Math.PI)) * Math.exp(- 0.5 * x);
        };

        // Value of i-th gaussian at x.
        function value_of(i, x) {
            x = (x - mean[i]);
            x *= x * precision[i];
            return weight[i] * norm[i] * Math.exp(- 0.5 * x);
        };

        // Value of all gaussians summed at x.
        function value(x) {
            var i, sx, g = 0;

            for (i = 0; i < N; ++i) {
                sx = x - mean[i];
                sx *= precision[i] * sx;
                g += weight[i] * norm[i] * Math.exp(- 0.5 * sx);
            }

            return g;
            //            return Math.max(1e-10, g);
        }

        // Removes those components whose weights fall below the given threshold.
        // The threshold is relative to the total weights of all the gaussians.
        function simplify(weight_threshold) {
            var i, abs_threshold = weight_threshold * this.sumOfWeights();

            for (i = 0; i < N;) {
                if (weight[i] < abs_threshold) {
                    remove(i); // Updates N automatically.
                } else {
                    ++i;
                }
            }

            return me;
        }

        // Normalizes the weights to 1.
        function normalize() {
            var i, wsum = 0;

            for (i = 0; i < N; ++i) {
                wsum += weight[i];
            }

            for (i = 0; i < N; ++i) {
                weight[i] /= wsum;
            }

            return me;
        }

        // Export list.
        this.length = N; // Keep track of number of gaussians in the model.

        this.weight     = function (i) { return weight[i]; };
        this.mean       = function (i) { return mean[i]; };
        this.precision  = function (i) { return precision[i]; };
        this.sigma      = function (i) { return 1 / Math.sqrt(precision[i]); };
        this.variance   = function (i) { return 1 / precision[i]; }; // = sigma ^ 2
        this.peak       = function (i) { return weight[i] * norm[i]; }

        this.add        = add;
        this.remove     = remove;
        this.prec_value_of = prec_value_of;
        this.value_of   = value_of;
        this.value      = value;
        this.simplify   = simplify;
        this.normalize  = normalize;
        this.instance_number = this.instance_counter;
        this.__proto__.instance_counter++;

        return this;
    }

    var GMM = GaussianMixtureModel; // Alias for typing convenience.

    // A class global property to keep track of number of instances.
    GaussianMixtureModel.prototype.instance_counter = 0;

    // Cloning a model.
    GaussianMixtureModel.prototype.clone = function () {
        var gmm = new GaussianMixtureModel();
        var i, len;
        for (i = 0, len = this.length; i < len; ++i) {
            gmm.add(this.weight(i), this.mean(i), this.precision(i));
        }
        return gmm;        
    };

    // Returns a plain array representation of the model.
    // Each element is one gaussian g specified as a 3-element
    // array. g[0] is the gaussian's weight, g[1] is its
    // mean and g[1] is its variance.
    GaussianMixtureModel.prototype.rep = function (f0) {
        f0 = f0 || 158.2;
        var rep = [];
        var i, m, s;
        for (i = 0; i < this.length; ++i) {
            m = this.mean(i);
            s = this.sigma(i);
            rep.push([declim[3](this.weight(i)), toSemitones(m/f0), toSemitones((m+s)/m)]);
        }

        return rep;
    };

    function toSemitones(freqRatio) {
        return declim[2](12 * Math.log(freqRatio) / Math.LN2);
    }

    // Convert to string for printing out.
    GaussianMixtureModel.prototype.toString = function (f0) {
        return JSON.stringify(this.rep(f0));
    };

    // Calculate the sum of the weights. Should usually be around 1.0.
    GaussianMixtureModel.prototype.sumOfWeights = function () {
        var i, len, w;
        for (i = 0, len = this.length, w = 0; i < len; ++i) {
            w += this.weight(i);
        }
        return w;
    };

    // Calculates a "normalized" peak value, taking all model
    // components into account. Useful for plotting the model.
    // The given fdx value is a step resolution to which you
    // need the peak calculated. This can be, for example,
    // the frequency difference between two adjacent pixels
    // in a graph where the x axis represents frequency and the
    // gaussian model's mean is in frequency units.
    GaussianMixtureModel.prototype.peakNorm = function (fdx) {
        var norm = 0;
        var wsum = 0;
        var x;
        var i, len;
        for (i = 0, len = this.length; i < len; ++i) {
            wsum += this.weight(i);
            x = 0.5 * fdx / this.sigma(i);
            norm += this.weight(i) * Math.log(gaussian_integral(x) - gaussian_integral(-x));
        }

        return Math.exp(- norm / wsum);
    };

    // Gives the peak value of the i-th gaussian. For description
    // of "fdx", see above "peakNorm" method.
    GaussianMixtureModel.prototype.peak_of = function (i, fdx) {
        var x = 0.5 * fdx / this.sigma(i);
        return this.weight(i) * (gaussian_integral(x) - gaussian_integral(-x));
    };

    // Calculates the probability of the random variable falling in the
    // interval [x1, x2] for the i-th gaussian, **NOT** taking the weight
    // of the gaussian into account.
    GaussianMixtureModel.prototype.probability_of = function (i, x1, x2) {
        x1 = (x1 - this.mean(i)) / this.sigma(i);
        x2 = (x2 - this.mean(i)) / this.sigma(i);
        return gaussian_integral(x2) - gaussian_integral(x1);
    };

    // Adds up all the probabilities for the given interval.
    GaussianMixtureModel.prototype.probability = function (x1, x2) {
        var i, len, p = 0;
        for (i = 0, len = this.length; i < len; ++i) {
            x1 = (x1 - this.mean(i)) / this.sigma(i);
            x2 = (x2 - this.mean(i)) / this.sigma(i);
            p += this.weight(i) * (gaussian_integral(x2) - gaussian_integral(x1));
        }

        return p;
    };

    // See http://www.ll.mit.edu/mission/communications/ist/publications/0802_Reynolds_Biometrics-GMM.pdf
    // for a description of the iteration method.
    //
    // This function gives the "a posteriori" probability for the component i, given x.
    // In other words, this is our guess that x is associated with the gaussian i.
    GaussianMixtureModel.prototype.aposteriori_probability = function (i, given_x) {
        return this.value_of(i, given_x) / this.value(given_x);
    };

    GaussianMixtureModel.prototype.aposteriori_probability2 = function (i, x1, x2) {
        return this.probability_of(i, x1, x2) / this.probability(x1, x2);
    };

    // The number of sub harmonics for "method of subharmonic reassignment".
    // According to this method, the sub-harmonic which yields the maximum
    // likelihood of all subharmonics that we search, for the purpose
    // of association with a single gaussian, is taken as the remapped
    // frequency value of a candidate frequency.
    //
    // If that is confusing to state, here is a breakdown -
    // 1. Consider a candidate frequency F. You want to find out how likely
    //    is it that this frequency is associated with a gaussian G.
    // 2. Calculate G(F), G(F/2), G(F/3) ....
    // 3. Whichever subdivision of F (say K) gives the maximum value in the series,
    //    remap F to F/K, so that in the expectation maximization iteration,
    //    the gaussian G will get biased towards F/K and the energy at F will
    //    get attributed to F/K.
    GaussianMixtureModel.prototype.number_of_subharmonics = 10;

    GaussianMixtureModel.prototype.value_of_subharmonic = function (i, f) {
        var k, val, maxn = 1, maxval = -1;
        var n = this.number_of_subharmonics;
        for (k = 1; k <= n; ++k) {
            val = this.value_of(i, f / k);
            if (maxval < val) {
                maxval = val;
                maxn = k;
            }
        }

        this.__sub_harmonic = f / maxn;
        return maxval;
    };

    GaussianMixtureModel.prototype.value_subharmonic = function (f) {
        var k, val, maxn = 1, maxval = -1;
        var n = this.number_of_subharmonics;
        for (k = 1; k <= n; ++k) {
            val = this.value(f / k);
            if (maxval < val) {
                maxval = val;
                maxn = k;
            }
        }

        this.__sub_harmonic = f / maxn;
        return maxval;
    };

    // Update the gaussian models based on the given spectrum.
    // Niter is the number of update iterations to do.
    // Give small numbers for Niter ... like 4 please!
    //
    // A new model is created with the updated gaussians.
    //
    // If any peaks need to be removed due to singularities,
    // the corresponding peaks in orig_model are also removed
    // for consistency. When this happens, the spectrum is
    // edited to smooth out the singularity.
    GaussianMixtureModel.prototype.iterate = function (spectrum, Niter, orig_model) {
        if (Niter <= 0) { 
            this.spectrum = spectrum;
            this.time_secs = spectrum.time_secs;
            return this; 
        }

        var i, j, N, len, f, rf, p, w, µ, σ, m, e, v, reiterate = false;
        var dfs = 0.001 * spectrum.df;
        var new_model = new GaussianMixtureModel();

        for (i = 0, N = this.length; i < N; ++i) {
            w = 0;
            µ = 0;
            σ = 0;
            e = 0;
            //m = model[i];
            for (j = 0, len = spectrum.length; j < len; ++j) {
                e += spectrum.energy[j];
                // We need to multiply pr_gaussian with the energy in order to
                // account for the fact that we aren't sampling frequency
                // values, but instead we're just running through the spectrum
                // and the "sampled" values are supposed to occur in proportion
                // to the energy.
                f = spectrum.frequency(j);
                rf = spectrum.reassigned_frequency(j);
                v = this.value_subharmonic(rf);
                if (v < 1e-5 * this.weight(i)) {
                    //                    debugger;
                    continue;
                }

                //                rf = this.__sub_harmonic;
                p = this.value_of_subharmonic(i, rf) / v;
                p *= spectrum.energy[j];
                rf = this.__sub_harmonic;
                w += p;
                µ += p * rf;
                σ += p * rf * rf;
            }

            if (isNaN(w)) { debugger; }

            if (w / (e * this.weight(i))  > 0.00001) {
                µ /= w;
                σ = Math.max(1e-10, (σ / w) - µ * µ);
                w /= e;
                new_model.add(w, µ, 1/σ);
            } else {
                //debugger;
            }
        }

        //        new_model.normalize();

        // Remove singular points in the components if we get component singularities.
        if (false) {
            for (i = 0, len = new_model.length, reiterate = false; i < len;) {
                if (new_model.sigma(i) < dfs) {
                    //debugger;
                    var index = Math.round((new_model.mean(i) - spectrum.fromFreq_Hz) / spectrum.df);
                    index = Math.max(0, Math.min(spectrum.length - 1, index));
                    if (index <= 0 || index >= spectrum.length - 1) {
                        spectrum.energy[index] = 0;
                    } else {
                        // Remove the sharp peak in the energy spectrum causing this singularity.
                        spectrum.energy[index] = 0.5 * (spectrum.energy[index - 1] + spectrum.energy[index + 1]);
                    }
                    new_model.remove(i);
                    this.remove(i);
                    if (orig_model && orig_model != this) { orig_model.remove(i); }
                    len = new_model.length;
                    reiterate = true;
                } else {
                    ++i;
                }
            }
        }

        //        return new_model.iterate(spectrum, reiterate ? Niter : (Niter - 1), orig_model);
        return new_model.iterate(spectrum, Niter - 1, orig_model);
    };

    // Returns a [-1,1] range number indicating how well this model
    // covers the given spectrum.
    //
    // The "coverage" is the correlation between the gaussian
    // mixture model and the energy spectrum. If the "coverage" is 1,
    // it means they are perfectly correlated. If -1, then they are 
    // perfectly negatively correlated (which doesn't happen for this case).
    GaussianMixtureModel.prototype.coverage = function (spectrum) {

        // The mixture model's 'spectrum' member holds the spectrum
        // which is modeled by it. So if one is not supplied, use this one.
        spectrum = spectrum || this.spectrum;

        var e, gval, i, len, gesum, e2sum, g2sum, gsum, esum, gm, g, f, freq, N, pN;
        var minf = spectrum.fromFreq_Hz, maxf = spectrum.toFreq_Hz;
        var df = spectrum.df;
        gesum = 0;
        g2sum = 0;
        e2sum = 0;
        gsum = 0;
        esum = 0;

        for (f = 0, N = spectrum.length; f < N; ++f) {
            e = spectrum.energy[f];
            freq = spectrum.frequency(f);
            gval = this.value(freq);
            g2sum += gval * gval;
            gsum += gval;
            gesum += gval * e;
            e2sum += e * e;
            esum += e;
        }

        gesum /= N;
        gsum /= N;
        esum /= N;
        g2sum /= N;
        e2sum /= N;

        var coverage = (gesum - gsum * esum) / Math.sqrt((e2sum - esum * esum) * (g2sum - gsum * gsum));
        return coverage;
    };

    // Returns the log-likelihood that the model fits the given spectrum.
    GaussianMixtureModel.prototype.likelihood = function (spectrum) {
        var likelihood = 0;
        var i, N = spectrum.length, f, esum = 0;
        for (f = 0; f < N; ++f) {
            likelihood += spectrum.energy[f] * Math.log(this.value(spectrum.frequency(f)));
            esum += spectrum.energy[f];
        }

        return likelihood / (esum * Math.LN2);
    };

    // Multiplies the i-th gaussian in GMM m1 with the i-th gaussian
    // in the GMM m2 and returns a new gaussian.
    //
    // Note that the product of two gaussian models is another
    // gaussian and obeys the following formula -
    //
    // g1 = [w1, µ1, σ1]
    // g2 = [w2, µ2, σ2]
    //
    // then G = g1 x g2 = [w, µ, σ] where
    // 1/σ = 1/σ1 + 1/σ2
    // <µ> = σ(µ1/σ1 + µ2/σ2)
    // <µ^2> = σ(µ1^2/σ1 + µ2^2/σ2)
    // w = {w1w2 / √(2π(σ1 + σ2))} x e^-((1/2σ)(<µ^2> - <µ>^2))
    //
    // Note: Remember that in this code, we use σ to denote the square of
    // the standard deviation.
    //
    GaussianMixtureModel.pairwise_product = function (m1, m2) {
        if (m1.length !== m2.length) {
            debugger;
            throw "Model order has to be the same for pairwise product.";
        }

        var g = new GaussianMixtureModel();
        var i, w, µ, µ1, µ2, µµ, p;
        for (i = 0; i < m1.length; ++i) {
            p = m1.precision(i) + m2.precision(i);
            µ1 = m1.mean(i);
            µ2 = m2.mean(i);
            µ = (µ1 * m1.precision(i) + µ2 * m2.precision(i)) / p;
            µµ = (µ1 * µ1 * m1.precision(i) + µ2 * µ2 * m2.precision(i)) / p;
            //          w = m1.weight(i) * m2.weight(i) / Math.sqrt(2 * Math.PI * (1 / m1.precision(i) + 1 / m2.precision(i)));
            w = m1.weight(i) * m2.weight(i) * Math.exp(- 0.5 * p * (µµ - µ * µ));

            g.add(w, µ, p);
        }

        return g;
    };

    // Creates a new model with N1 x N2 entries in it corresponding
    // to each pair of gaussians in the two models. The index of the
    // product of (i,j) is given by i * N2 + j.
    //
    // The tolerance is a value of the minimum spread (in the same units
    // as the mean, i.e. it is the statistical sigma) that has to be used 
    // in the outer product calculation. This is useful when the gaussians are
    // mostly very sharp and therefore yield absurdly small weights when
    // multiplied with even small shifts. The tolerance value, therefore,
    // indicates how much shift can you "tolerate" when considering
    // whether two peaks are "related".
    //
    GaussianMixtureModel.outer_product = function (m1, m2, tolerance) {
        tolerance = tolerance ? tolerance : 1e-5;

        var g = new GaussianMixtureModel();
        var maxprec = 1 / (tolerance * tolerance);


        var normfactor = Math.sqrt(GMM.relatedness(m1, m1, tolerance) * GMM.relatedness(m2, m2, tolerance));

        var i, j, N1, N2, w, w1, µ, lµ, µ1, µ2, µµ, lµµ, p, p1, p2, lp, lp1, lp2;
        for (i = 0, N1 = m1.length; i < N1; ++i) {
            p1 = m1.precision(i);
            lp1 = Math.min(maxprec, p1);
            µ1 = m1.mean(i);
            w1 = m1.weight(i);
            for (j = 0, N2 = m2.length; j < N2; ++j) {
                p2 = m2.precision(j);
                lp2 = Math.min(maxprec, p2);
                lp = lp1 + lp2;
                µ2 = m2.mean(j);
                lµ = (µ1 * lp1 + µ2 * lp2) / lp;
                lµµ = (µ1 * µ1 * lp1 + µ2 * µ2 * lp2) / lp;
                //          w = m1.weight(i) * m2.weight(i) / Math.sqrt(2 * Math.PI * (1 / m1.precision(i) + 1 / m2.precision(i)));
                w = w1 * m2.weight(j) * Math.exp(- 0.5 * lp * (lµµ - lµ * lµ));

                g.add(w / normfactor, (µ1 * p1 + µ2 * p2) / (p1 + p2), p1 + p2);
            }
        }

        return g;
    };


    // Returns a value in the range [0,1] indicating how related
    // the two given gaussian mixture models are, given some tolerance.
    //
    // Note that this function is practically identical to the "outer_product"
    // function above and can actually be equivalently written as -
    //
    // return (GaussianMixtureModel.outer_product(m1, m2, tolerance).sumOfWeights() / (m1.sumOfWeights() * m2.sumOfWeights());
    // 
    // .. but doing it that way would unnecessarily create a new GMM
    // model object and immediately throw it away after getting the sum of weights.
    GaussianMixtureModel.simple_relatedness = function (m1, m2, tolerance) {

        tolerance = tolerance ? tolerance : 1e-5;
        var maxprec = 1 / (tolerance * tolerance);
        var i, j, N1, N2, w1, µ, µ1, µ2, µµ, p, p1, p2, w, wa, wb, wanorm = 0, wbnorm = 0, wsum = 0, wnorm = 0;

        for (i = 0, N1 = m1.length; i < N1; ++i) {
            p1 = Math.min(maxprec, m1.precision(i));
            µ1 = m1.mean(i);
            w1 = m1.weight(i);
            for (j = 0, N2 = m2.length; j < N2; ++j) {
                p2 = Math.min(maxprec, m2.precision(j));
                p = p1 + p2;
                µ2 = m2.mean(j);
                µ = (µ1 * p1 + µ2 * p2) / p;
                µµ = (µ1 * µ1 * p1 + µ2 * µ2 * p2) / p;
                wa = w1 * m2.weight(j);
                wb = Math.exp(- 0.5 * p * (µµ - µ * µ));
                wsum += wa * wb;
                wnorm += wa;
            }
        }

        wsum /= wnorm;

        return wsum;
    };

    // Close to simple_relatedness, but normalizes along the way.
    GaussianMixtureModel.relatedness = function (m1, m2, tolerance) {
        if (!m1.__self_relatedness) {
            m1.__self_relatedness = GaussianMixtureModel.simple_relatedness(m1, m1, tolerance);
        }

        if (!m2.__self_relatedness) {
            m2.__self_relatedness = GaussianMixtureModel.simple_relatedness(m2, m2, tolerance);
        }

        return GaussianMixtureModel.simple_relatedness(m1, m2, tolerance) / Math.sqrt(m1.__self_relatedness * m2.__self_relatedness);
    };

    (function (x, y) {
        x = new GaussianMixtureModel();
        x.add(1/2,100,1).add(1/4,200,2).add(1/4,500,5);
        debugger;
        y = GaussianMixtureModel.relatedness(x,x);
        console.log(y);
    })();

    // Makes an initial guess at the peaks of this spectrum.
    // Returns an array of gaussians.
    function guess_peaks(spectrum) {
        // Find all the peaks in the scan range.
        var len, a, µ, σ, e, s1, s2, s3, f0, df, f1, f2, f3;
        var energy = spectrum.reassigned_energy;
        var model = new GaussianMixtureModel();

        for (i = 1, len = spectrum.length - 1, f0 = spectrum.fromFreq_Hz, df = spectrum.df; i < len; ++i) {
            s1 = energy[i];
            s2 = energy[i-1];
            s3 = energy[i+1];
            f1 = f0 + i * df;
            f2 = f0 + (i - 1) * df;
            f3 = f0 + (i + 1) * df;
            if (s1 > s2 && s1 > s3) {
                // We have a local peak.
                a = energy[i];// / spectrum.power;
                e = Math.max(0.0001, s1 + s2 + s3);
                µ = (s1 * f1 + s2 * f2 + s3 * f3) / e;
                σ = (s1 * f1 * f1 + s2 * f2 * f2 + s3 * f3 * f3) / e - µ * µ;
                model.add(a, µ, 1/σ);
            }
        }

        // Store away the spectrum in the peaks array itself.
        model.spectrum = spectrum;
        model.time_secs = spectrum.time_secs;

        return model;
    };

    // Given an array of filters to apply to this spectrum,
    // returns a new array of filtered spectra. The spectrum 
    // needs to have been initialized with a measure() call.
    function guess_filtered_peaks(spectrum, filters, lpf_iter) {
        return filters.map(function (f) {
            return guess_peaks(spectrum.clone().lpf(lpf_iter).filter(f));
        });
    }

    // DINOSAUR
    function find_initial_peaks_around_hz(fromFreq_Hz, toFreq_Hz, measure, time_secs) {
        // Calc all the components in the scan range.
        //debugger;
        var spec = new Spectrum(fromFreq_Hz, toFreq_Hz, measure, time_secs);
        var spec = (new Spectrum()).setup(measure.buffer.sampleRate, Math.floor(measure.buffer.sampleRate * measure.window_secs), fromFreq_Hz, toFreq_Hz);
        spec.samples = measure.samples;
        var filtered_spectrum = spec.update(time_secs).clone().lpf(k_lpf_iterations);
        var peaks = guess_peaks(filtered_spectrum);

        if (peaks.length === 0 && (toFreq_Hz < fromFreq_Hz * Math.pow(2,10/12))) {
            // Expand the window by a semitone (an extra quartertone lower and higher)
            // and try again. We don't do this beyond a fourth around the scan
            // frequency.
            return find_initial_peaks_around_hz(fromFreq_Hz * Math.pow(2, -0.25/12), toFreq_Hz * Math.pow(2, 0.25/12), measure, time_secs);
        } else {
            return peaks; // peaks.spectrum is the spectrum which contains all the other info you need.
        }
    }

    // DINOSAUR
    // Function to limit the number of decimal places of the gaussian model
    // parameters for presentation purposes.
    function declim_gaussian_models(ms) {
        return ms.map(function (m) { return [declim[3](m[0]), declim[1](m[1]), declim[1](m[2]), declim[3](m[3])]; });
    }

    // DINOSAUR
    // Note that in this formula, "σ" actually refers to the square `
    // of the standard deviation. Since we usually deal with the squared
    // value everywhere, I just decided to avoid the squaring altogether
    // and use the squared value itself everywhere. Even the iteration
    // for the gaussian mixture model computes the squared value only.
    function gaussian(x, w, µ, σ) {
        var s = Math.max(0.00001, σ);
        return w * Math.exp(- (x - µ) * (x - µ) / (2 * s)) / Math.sqrt(2 * Math.PI * s);
    }

    function decreasing_weight_order(a, b) { return b[0] - a[0]; }
    function increasing_spread_order(a, b) { return a[2] - b[2]; }

    // For each peak, gives a probability value that the peak has
    // been "tracked" in this time interval. For each peak, three
    // numbers are given - the first one being the probability of
    // tracking (expressed in dB) and the second one giving the change in frequency 
    // expressed in cents. The third number is a precision of the
    // peak expressedin cents. This is = centre frequency / variance 
    // and is also known as "Q factor" and is expressed in octaves.
    GMM.tracking_probabilities = function (peaks_before, peaks_after) {
        var i, len, prod, p, df, q, probabilities = [];

        var ps = new Float64Array(peaks_before.length);
        var dfs = new Float64Array(peaks_before.length);
        var qs = new Float64Array(peaks_before.length);

        var prod = GMM.pairwise_product(peaks_before, peaks_after);

        for (i = 0, len = peaks_before.length; i < len; ++i) {
            // Limit the decimal places here itself. We aren't interested
            // in what happens beyond the fifth one!
            ps[i] = declim[3](10 * Math.log(prod.weight(i) / (peaks_before.weight(i) * peaks_after.weight(i))) / Math.log(10));

            // Measure the change in frequency in cents.
            dfs[i] = Math.round(1200 * Math.log(peaks_after.mean(i) / peaks_before.mean(i)) / Math.LN2);

            // Calculate tracking q factor of the peak.
            qs[i] = Math.round(10 * Math.log(prod.mean(i) / prod.sigma(i)) / Math.LN2) / 10;
        }

        return {
            length: ps.length,
                probability_db: ps,
                df_st: dfs,
                q_db: qs
        };
    };

    // This one determines, on a peak-by-peak basis,
    // whether it has converged, and if so returns the
    // sum of weights of all those peaks that have converged.
    //
    // dfTh is the frequency difference threshold below which
    // both the centre frequency as well as the spread have to
    // fall for "convergence" to occur. It is expressed in 
    // octaves. 0.0005 is a reasonable value to use.
    GMM.convergence = function (peaks_before, peaks_after, dfTh) {
        var i, len, wsum = 0, wtotal = 0, df1, df2;
        dfTh = dfTh || 0.0005;
        for (i = 0, len = peaks_before.length; i < len; ++i) {
            wtotal += peaks_after.weight(i);
            df1 = Math.log(peaks_after.mean(i) / peaks_before.mean(i)) / Math.LN2;
            df2 = Math.log(peaks_after.sigma(i) / peaks_after.sigma(i)) / Math.LN2;
            if (Math.abs(df1) < dfTh && Math.abs(df2) < dfTh) {
                // This peak has converged.
                wsum += peaks_after.weight(i);
            }
        }

        return wsum / wtotal;
    };

    // Iterates the model until the "convergence" measure
    // crosses the given threshold. The return value is
    // the result of the most recent iteration.
    //
    // The orig_model is passed in separately so that it 
    // can be kept up to date if the iterations end up
    // removing singularities.
    //
    // model and orig_model are GaussianMixtureModel objects
    // spectrum is a Spectrum object.
    function wait_for_convergence(spectrum, model, orig_model, threshold) {
        var i, m1 = model, m2 = model, convergence = 0;
        while (m1.length > 0 && m2.length > 0 && convergence < threshold) {
            m1 = m2;
            m2 = m1.iterate(spectrum, 2, orig_model);
            if (m1.length === m2.length) {
                convergence = GMM.convergence(m1, m2);
                elem('gmm_stability').innerHTML = declim[3](convergence);
            }
        }

        return m2;
    }

    // Attempt at a FINAL "stabilization algorithm".
    // The key is the GMM.convergence function which is used to
    // keep track of the portions of the model that have converged
    // during iterations.
    //
    // 1. First, iterations are run until the convergence crosses
    //    a threshold of 0.5.
    // 2. After 0.5 of the model converges, the model is simplified
    //    by removing those gaussians with weights below 0.01.
    // 3. Iterations are continued with the simplified model until
    //    convergence crosses 0.9, at which point the model is
    //    declared to have converged.
    //
    // When params.track_peaks is set to true, then the routine
    // will not try to simplify the model.
    //
    // Returns the stabilized model.
    GMM.stabilize = function (spectrum, model, params) {
        // params can be used to customize the stabilization thresholds.
        var simplify_convergence_threshold = (params && params.simplify_convergence_threshold) || 0.5;
        var simplify_weight_threshold = (params && params.simplify_weight_threshold) || 0.01;
        var convergence_threshold = (params && params.convergence_threshold) || 0.9;
        var track_peaks = (params && ('track_peaks' in params)) ? params.track_peaks : false;

        var convergence = 0;
        var m1 = model, m2 = model, orig_model = model;
        var i, len, N = 4; // N is the number of iterations done in each batch.

        if (true) {
            // Wait until some of the peaks have converged.
            m2 = wait_for_convergence(spectrum, m1, orig_model, simplify_convergence_threshold);

            // Simplify the model.
            if (track_peaks === false) {
                m2.simplify(simplify_weight_threshold);
                // TODO: Need to update m1 as well?
            }

            // Do the final batch of iterations.
            m1 = m2;
            m2 = wait_for_convergence(spectrum, m1, orig_model, convergence_threshold);
        } else {
            m2 = m1.iterate(spectrum, 1, orig_model);

            if (track_peaks === false) {
                m2.simplify(simplify_weight_threshold);
                // TODO: Need to update m1 as well?
            }
        }

        return m2;
    };

    function frac2db(f) {
        return 10 * Math.log(f) / Math.log(10);
    }

    // Runs a sliding window maxmin filter on the given
    // vector of numbers with a window width of W. 
    // Returns a new vector with the filtered result.
    //
    // The vector must at least be of length 2.
    function morphop(vec, op) {

        var result = [];

        result[0] = op(vec[0], vec[1]);
        result[vec.length - 1] = op(vec[vec.length - 2], vec[vec.length - 1]);

        var i, N;
        for (i = 1, N = vec.length - 1; i < N; ++i) {
            result[i] = op(vec[i-1], vec[i], vec[i+1]);
        }

        result.fn_erode = vec.fn_erode;
        result.fn_dilate = vec.fn_dilate;

        return result;
    }

    morphop.dilate = function (vec, N) {
        if (N <= 0) {
            return vec;
        } else {
            return morphop.dilate(morphop(vec, vec.fn_dilate), N - 1);
        }
    };

    morphop.erode = function (vec, N) {
        if (N <= 0) {
            return vec;
        } else {
            return morphop.erode(morphop(vec, vec.fn_erode), N - 1);
        }
    };

    morphop.opening = function (vec, N) {
        return morphop.dilate(morphop.erode(vec, N), N);
    };

    morphop.closing = function (vec, N) {
        return morphop.erode(morphop.dilate(vec, N), N);
    }

    // "frequencies" is an array of Hz values. This function will return an array of the same length as
    // "frequencies", with each entry giving a time track of a likelihood value of a particular time t
    // featuring that pitch. The arrays will have N entries where N = (toTime_secs - fromTime_secs) / step_secs.
    // All these arrays have a .time(i) method which gives the time corresponding to the given index.
    // 
    // params is an optional object using which you can give additional control parameters for the
    // tracking. 
    //    - 'harmonics' = integer giving the number of harmonics to track for each given frequency, defaults to 5.
    GMM.track_pitches = function (measure, frequencies, fromTime_secs, toTime_secs, window_secs, step_secs, params) {
        var NHarmonics = (params && params.harmonics) || 5;
        var spread_st = (params && params.spread_semitones) || 1;
        var N = Math.ceil((toTime_secs - fromTime_secs) / step_secs);

        function time(i) {
            return fromTime_secs + i * step_secs;
        };

        // Create the frequencies' tracking probability arrays.
        var tracks = (function (i, f, tracks) {
            for (i = 0, tracks = []; i < frequencies.length; ++i) {
                tracks.push(new Float64Array(N));
            }

            return tracks;
        })();

        // Create a GMM for the harmonics of each frequency.
        var harmonic_gmms = (function (i, f, g, h, w, p, q, gmms) {
            for (i = 0, w = 1 / NHarmonics, gmms = []; i < frequencies.length; ++i) {
                f = frequencies[i];
                g = new GaussianMixtureModel();
                q = f * (Math.pow(2, spread_st / 12) - 1);
                p = 1 / (q * q);
                for (h = 1; h <= NHarmonics; ++h) {
                    g.add(w, f * h, p/(h*h));
                }
                gmms.push(g);
            }

            return gmms;
        })();

        var gmms = []; // An array to store the GMM objects for each time step.

        // Track each time step and store the relatedness values of the harmonics.
        var i, f, spectrum, peaks, continuity = [0];

        // The lower end of the range we need is an octave below the lowest frequency.
        // The upper end should at least account for the highest harmonic we need to
        // examine.
        var fromFreq_Hz = Math.min.apply(null, frequencies) / 2;
        var toFreq_Hz = Math.max.apply(null, frequencies) * (NHarmonics + 1);
        var samples = measure.buffer.getChannelData(0);
        var sampleRate = measure.buffer.sampleRate;

        // We reuse this one spectrum object throughout the tracking.
        spectrum = (new Spectrum()).setup(sampleRate, Math.floor(sampleRate * window_secs), fromFreq_Hz, toFreq_Hz);
        spectrum.samples = samples;

        for (i = 0; i < N; ++i) {
            // Measure for the current time.
            spectrum.update(time(i));

            // Guess at the spectral peaks after doing a spectrum low pass.
            // the spectrum low pass is equivalent to smooth windowing in the
            // time domain.
            peaks = guess_peaks(spectrum.lpf(k_lpf_iterations));

            // Iterate the gaussians and stabilize the model based on dual
            // information from both the power spectrum as well as the frequency
            // reassignment information. Note that the lpf above only affects
            // the power spectrum and not the frequency reassignment.
            peaks = GMM.stabilize(spectrum, peaks);
            peaks.power = spectrum.power;

            // Store away the gmm object for this time slice.
            gmms.push(peaks);

            // The tracking probability for each harmonic is simply the 
            // "relatedness" value of this gaussian model with the 
            // gaussian model of the harmonics.
            for (f = 0; f < tracks.length; ++f) {
                tracks[f][i] = GMM.relatedness(peaks, harmonic_gmms[f]);
            }
        }

        gmms.fn_erode = function (g1, g2, g3) {
            if (g3) {
                var r12 = GMM.relatedness(g1, g2, 10);
                var r23 = GMM.relatedness(g2, g3, 10);
                return r12 < r23 ? g1 : g3;
            } else {
                return g2;
            }
        };

        gmms.fn_dilate = function (g1, g2, g3) {
            if (g3) {
                var r12 = GMM.relatedness(g1, g2, 10);
                var r23 = GMM.relatedness(g2, g3, 10);
                return r12 > r23 ? g1 : g3;
            } else {
                return g2;
            }
        }

        debugger;
        //gmms = morphop.opening(gmms, 5);
        debugger;

        for (i = 1; i < gmms.length; ++i) {
            continuity.push(GMM.relatedness(gmms[i], gmms[i-1], 10));
        }

        return {
            tracks: tracks,
                continuity: continuity,
                peaks: gmms,
                time: time,
                fromTime_secs: fromTime_secs,
                toTime_secs: toTime_secs,
                step_secs: step_secs
        };
    };

    // Takes a peaks array - an N-length array of gaussian mixture models - and
    // returns a self similarity matrix as a Float32Array of length N x N where 
    // the value at (i,j) = i x N + j gives the similarity between peaks[i] and 
    // peaks[j]. Since this is a "self similarity" measure, the value at (i,j)
    // is the same as the value at (j, i). Expect all the diagonal values to 
    // be = 1.0.
    GMM.self_similarity = function (peaks, tolerance) {
        var N = peaks.length;
        var i, j, pi;
        var ss = new Float32Array(N * N);
        for (i = 0; i < N; ++i) {
            pi = peaks[i];
            for (j = 0; j <= i; ++j) {
                ss[i * N + j] = ss[j * N + i] = GMM.relatedness(pi, peaks[j], tolerance);
            }
        }
        return ss;
    };

    // DINOSAUR
    // TODO:
    GMM.track = function (measure, fromTime_secs, toTime_secs, window_secs, step_secs, params) {
        var tracking_break_threshold_db = (params && params.tracking_break_threshold_db) || -3.0;
        var coverage_change_threshold_db = (params && params.coverage_change_threshold_db) || -1.5;

        var t = fromTime_secs, t_end = toTime_secs - window_secs;
        var do_peak_tracking = { track_peaks: true };
        var spectra = [null, null];
        var spectrum_ix = 0;
        var spectrum;
        var tracked_peaks = [], peak_set, peaks, model, tracking_info, coverage_before, coverage_after, spectrum;
        var i, len, stats, w1, w2, stats_i, µ, σ, p, n, pt, pf;
        var now_at = elem('now_at');

        function tracker() {
            if (t >= t_end) {
                return null;
            }

            if (!peak_set) {
                // Do an initial guess at the peaks.
                spectra[spectrum_ix] = spectrum = new Spectrum(refFreqLow_Hz, refFreqHigh_Hz, measure, t);
                peaks = guess_peaks(spectrum.clone().lpf(k_lpf_iterations));
                peaks.spectrum = spectrum;

                // Stabilize the peaks.
                peaks = GMM.stabilize(spectrum, peaks); 

                peak_set = {time: t, peaks: peaks, coverage: peaks.coverage(spectrum)};
                tracked_peaks.push({
                    initial_peaks: peak_set, 
                    peaks: []
                });
            } else {
                // We don't need to do the peak estimation pass. We only
                // need to compute the spectrum.

                if (spectra[spectrum_ix]) {
                    spectrum = spectra[spectrum_ix].update(t);
                } else {
                    spectrum = spectra[spectrum_ix] = new Spectrum(refFreqLow_Hz, refFreqHigh_Hz, measure, t);
                }

                // Calculate the coverage of the previous peak set with this new spectrum
                // before tracking.
                coverage_before = peak_set.peaks.coverage(spectrum);

                // Track the previous peak set.
                peaks = GMM.stabilize(spectrum, peak_set.peaks, do_peak_tracking);

                // Calculate coverage after tracking.
                coverage_after = peaks.coverage(spectrum);

                // Calculate tracking measures.
                tracking_info = GMM.tracking_probabilities(peak_set.peaks, peaks);

                // Each peak has the following stats for the tracking probability
                // as well as the frequency shift.
                //
                // mean, mean of sqaure, and update weight of tracking probability in dB
                // mean, mean of square and update weight of peak shift in cents.
                // These are stored as -
                // [[µT, µµT, wT], [µP, sP, wP]]
                var track = tracked_peaks[tracked_peaks.length - 1];
                if (!track.tracking_stats) {
                    // If any of the tracking probabilities are below the
                    // threshold, discard this track and restart from next time.
                    var bad_track = false;
                    for (i = 0, len = tracking_info.length; i < len; ++i) {
                        if (tracking_info.probability_db[i] < tracking_break_threshold_db) {
                            bad_track = true;
                            break;
                        }
                    }

                    if (bad_track) {
                        tracked_peaks.pop();
                        peak_set = null;
                    } else {
                        // Valid track. Update the tracking stats.
                        stats = [];
                        for (i = 0, len = tracking_info.length; i < len; ++i) {
                            pt = tracking_info.probability_db;
                            pf = tracking_info.df_st;
                            stats.push([[pt[i], pt[i] * pt[1], 1], [pf[i], pf[i] * pf[i], 1]]);
                        }

                        peak_set = {time: t, peaks: peaks, coverage: coverage_after};
                        track.peaks.push(peak_set);
                        track.tracking_stats = stats;
                    }
                } else {
                    // Tracking stats exist. Update the stats using the new tracking info.
                    stats = track.tracking_stats;
                    for (i = 0, len = tracking_info.length; i < len; ++i) {
                        stats_i = stats[i];
                        stats_i[0][2] += 1;
                        stats_i[1][2] += 1;
                        w1 = 1 - 1 / stats_i[0][2];
                        w2 = 1 - 1 / stats_i[1][2];
                        stats_i[0][0] = stats_i[0][0] * w1 + tracking_info.probability_db[i] * (1 - w1);
                        stats_i[0][1] = stats_i[0][1] * w1 + tracking_info.probability_db[i] * tracking_info.probability_db[i] * (1 - w1);
                        stats_i[1][0] = stats_i[1][0] * w2 + tracking_info.df_st[i] * (1 - w2);
                        stats_i[1][1] = stats_i[1][1] * w2 + tracking_info.df_st[i] * tracking_info.df_st[i] * (1 - w2);
                    }

                    // Make a gaussian model out of the tracking stats and check whether
                    // the new tracking info is too unlikely.
                    for (i = 0, len = tracking_info.length, n = 0; i < len; ++i) {
                        stats_i = stats[i];
                        µ = stats_i[0][0];
                        σ = Math.max(0.25, stats_i[0][1] - µ * µ);
                        p = gaussian(tracking_info.probability_db[i], Math.sqrt(2 * Math.PI * σ), µ, σ);
                        µ = stats_i[1][0];
                        σ = Math.max(500, stats_i[1][1] - µ * µ);
                        p *= gaussian(tracking_info.df_st[i], Math.sqrt(2 * Math.PI * σ), µ, σ);

                        n += p * peaks.weight(i);  // Weight the tracking support probability by the 
                        // gaussian's own weight so that the most important
                        // ones get the attention.
                    }

                    if (frac2db(n) < tracking_break_threshold_db) {
                        // Tracking mostly broken looks like.
                        // Start a new peak set.
                        debugger;
                        peak_set = null;  
                        console.log("Tracked for " + declim[3](t - track.initial_peaks.time) + " secs");
                    } else {
                        // Add this peak set to the previous track.
                        peak_set = {time: t, peaks: peaks, coverage: coverage_after};
                        track.peaks.push(peak_set);
                    }
                }
            }

            t += step_secs;
            spectrum_ix = (spectrum_ix + 1) % spectra.length;
            return tracked_peaks;
        }

        return tracker;
    };

    var svaras_to_numbers_map = {
        S:0, r:1, R:2, g:3, G:4, m:5, M:6, P:7, d:8, D:9, n:10, N:11,
        sa:0, ri1:1, ri2:2, ri3:3, ga1:2, ga2:3, ga3:4, ma1:5, ma2:6, pa:7, da1:8, da2:9, da3:10, ni1:9, ni2:10, ni3:11,
        '+':12, '-':-12
    };

    var svara_pattern = /(sa|ri1|ri2|ri3|ga1|ga2|ga3|ma1|ma2|pa|da1|da2|da3|ni1|ni2|ni3|S|r|R|g|G|m|M|P|d|D|n|N|\+|\-)/g;

    function adder(a,b) { return a + b; }
    function sum(array) { return array.reduce(adder); }
    function asfn(mapper) { return function (key) { return mapper[key]; }; }

    // Convert string representation of svara sequence into an array of pitches
    // expressed in semitones. A "Svara sequence" is a space separated sequence
    // of tokens of the form "<svara_name>(<octave_indicator>*)".
    function svaras_to_pitches(svaras_str) {
        // Convert 'P D n S+' format to [7,9,11,12] format.

        // First split upon white space. Trim leading and trailing white spaces.
        var items = svaras_str.replace(/^(\s)+/, '').replace(/(\s)+$/, '').replace(/(\s)+/g, ' ').split(' ');

        if (!items) { return []; }

        var svaras_to_numbers = asfn(svaras_to_numbers_map);

        // Examine the '+' or '-' suffix and turn svara name 'ga3++' into 4+12+12 = 28
        return items.map(function (svs) { 
            var m = svs.match(svara_pattern);
            var n = m.map(svaras_to_numbers);
            return sum(n);
        });
    }

    // Return [fromFreq_Hz, toFreq_Hz] that encompasses all the pitches
    // in the given pitches array - given in semitones relative to the
    // given refFreq_Hz.
    function pitch_range(pitches, refFreq_Hz) {
        if (!pitches || pitches.length === 0) { 
            return [refFreq_Hz, refFreq_Hz]; 
        }

        // Find min and max pitches.
        var minPitch = Math.min.apply(null, pitches);
        var maxPitch = Math.max.apply(null, pitches);

        // Extend the range by a tone on either side.
        minPitch -= 1;
        maxPitch += 1;

        // Convert from semitones to Hz.
        return [minPitch, maxPitch].map(function (p) { return refFreq_Hz * Math.pow(2, p/12); });
    }




    // subharmonic(N) is a function that takes a gaussian
    // and returns its N subharmonic. The frequency of the
    // output gaussian is 1/N of the frequency of the input
    // gaussian.
    var subharmonic = (function () {
        var shifters = [];
        // Cached harmonic shifter closures.
        return function (N) {
            return shifters[N] || (shifters[N] = function (p) { 
                var g = [p[0], p[1]/N, p[2]/(N*N), 0];
                g[3] = gaussian(g[1], g[0], g[1], g[2]);
                return g;
            });
        };
    })();


    // Makes gaussian mixture models a small spread around freq_Hz, for N harmonics, for each window
    // in the time interval [fromTime_secs, toTime_secs], stepping by step_secs.
    //
    // audioBuffer is an object of type AudioBuffer (part of Web Audio API).
    //
    // The return value is an array of objects, each of which provides harmonic
    // information about the given frequency component at a time t. Each object
    // has two fields - 'time' and 'harmonics'. The first gives the time at which
    // the harmonics were modeled and the second is an array of models, one for
    // each harmonic, with the fundamental at index = 0, first octave at index = 1,
    // etc. (i.e. harmonic number = index + 1). The model is a standard model
    // object as returned by the find_peaks_around() function.
    function harmonic_modelgram(fromFreq_Hz, toFreq_Hz, N1, N2, audioBuffer, fromTime_secs, toTime_secs, window_secs, step_secs) {
        // Limit the time range according to what is available.
        fromTime_secs = Math.max(0, Math.min(audioBuffer.duration - window_secs, fromTime_secs));
        toTime_secs = Math.max(fromTime_secs + step_secs, Math.min(audioBuffer.duration, toTime_secs));
        if (fromTime_secs + step_secs > toTime_secs) {
            // Null time interval.
            return [];
        }

        var channelData = audioBuffer.getChannelData(0);
        var sampleRate = audioBuffer.sampleRate;
        var measure = function (f, t) {
            return periodicity(channelData, sampleRate, window_secs, f, t);
        };


        var harmonic, harmonic_model, modelgram = [], spread_Hz;
        var i, len, t;
        // We keep the Hz spread constant for the higher harmonics
        // as well so that we can get greater precision for them.
        var centreFreq_Hz = Math.sqrt(fromFreq_Hz * toFreq_Hz);
        var spread_Hz = Math.max(centreFreq_Hz - fromFreq_Hz, toFreq_Hz - centreFreq_Hz);
        for (t = fromTime_secs; t < toTime_secs; t += step_secs) {
            harmonic_model = null;
            for (i = N1; i < N2; ++i) {
                // Note that the spread of frequencies keeps expanding with N.
                // An alternative is to keep the spread fixed, but I'm not sure
                // how psychoacoustically valid that is.
                harmonic = find_peaks_around_hz(fromFreq_Hz * (i+1), toFreq_Hz * (i+1), measure, t);
                //                harmonic = find_peaks_around_hz(centreFreq_Hz * (i+1) - spread_Hz, centreFreq_Hz * (i+1) + spread_Hz, measure, t);                

                if (harmonic.peaks.length > 0) {
                    if (harmonic_model) {
                        harmonic_model.peaks = simplify_gaussians(gmm_product(harmonic_model.peaks, harmonic.peaks.map(subharmonic(i+1))).model);
                        harmonic_model.relative_energy += harmonic.relative_energy;
                    } else {
                        harmonic_model = harmonic;
                        harmonic_model.peaks = harmonic.peaks.map(subharmonic(i+1));
                    }
                } else {
                    if (!harmonic_model) {
                        harmonic_model = harmonic;
                    }
                }
            }

            harmonic_model.time = t;
            modelgram.push(harmonic_model);
        }

        return modelgram;
    }


    // Note that the product of two gaussian models is another
    // gaussian and obeys the following formula -
    //
    // g1 = [w1, µ1, σ1]
    // g2 = [w2, µ2, σ2]
    //
    // then G = g1 x g2 = [w, µ, σ] where
    // 1/σ = 1/σ1 + 1/σ2
    // <µ> = σ(µ1/σ1 + µ2/σ2)
    // <µ^2> = σ(µ1^2/σ1 + µ2^2/σ2)
    // w = {w1w2 / √(2π(σ1 + σ2))} x e^-((1/2σ)(<µ^2> - <µ>^2))
    //
    // Note: Remember that in this code, we use σ to denote the square of
    // the standard deviation.
    //
    function multiply_gaussians(g1, g2) {
        var w1 = g1[0], µ1 = g1[1], σ1 = g1[2];
        var w2 = g2[0], µ2 = g2[1], σ2 = g2[2];

        var σ = σ1 * σ2 / (σ1 + σ2);
        var µ = σ * (µ1 / σ1 + µ2 / σ2);
        var µµ = σ * (µ1 * µ1 / σ1 + µ2 * µ2 / σ2);
        //        var w = (w1 * w2 / Math.sqrt(2 * Math.PI * (σ1 + σ2))) * Math.exp(- 0.5 * (µµ - µ * µ) / σ);
        var w = w1 * w2 * Math.exp(- 0.5 * (µµ - µ * µ) / σ);

        return [w, µ, σ, gaussian(µ, w, µ, σ)];
    }

    function geometric_mean_of_gaussians(g1, g2) {
        var g = multiply_gaussians(g1, g2);
        g[0] = g[0] / Math.sqrt(g1[0] * g2[0]);
        g[2] = 2 * g[2];
        g[3] = gaussian(g[1], g[0], g[1], g[2]);
        return g;
    }

    // Product of two gaussian mixture models is another
    // gaussian mixture model. Given two GMMs of order
    // M and N respectively, the result is a matrix with
    // first index in range [0,M) and second index in
    // range [0,N), giving the product of the individual
    // gaussians.
    //
    // The result object has two fields - 
    // 'model' gives the result as another gaussian mixture model,
    // i.e. as a single array of length MxN,
    // and 'matrix' gives the result as an MxN matrix.
    function gmm_product(gmm1, gmm2) {
        var i, j, len1, len2, g1, g2, r = [], m, g;

        len1 = gmm1.length;
        len2 = gmm2.length;
        if (len1 === 0) { return {matrix: [gmm2], model: gmm2}; }
        if (len2 === 0) { return {matrix: [gmm1], model: gmm1}; }

        for (i = 0; i < len1; ++i) {
            g1 = gmm1[i];
            for (j = 0, g = []; j < len2; ++j) {
                g.push(multiply_gaussians(g1, gmm2[j]));
            }
            r.push(g);
        }

        return {
            matrix: r,
                get model() {
                    // Compute lazily and cache.
                    return m || (m = [].concat.apply([], r));
                }
        };
    }

    // Takes an array of gaussian model parameters of the form
    // [w, µ, σ, peak] and returns a new array where the weights
    // have all been normalized such that the sum of weights
    // of all the gaussians is unity.
    //
    // The return value is an object with two fields - 
    // 'wsum' which gives the sum of the original weights
    // and 'models' which gives the new array of models.
    function normalize_gaussian_weights(gs) {
        var wsum = 0;
        var i, len;
        for (i = 0, len = gs.length; i < len; ++i) {
            wsum += gs[i][0];
        }

        return {
            wsum: wsum,
                models: gs.map(function (g) { 
                    var ng = g[0] / wsum;
                    return [ng, g[1], g[2], gaussian(g[1], ng, g[1], g[2])];
                })
        };
    }

    var scanner = {
        buffer:         null,
        frequency_Hz:   refFreq_Hz,
        window_secs:    0.025,
        step_frac:      0.5,
        spread_factor:  0.5,
        time:           0.0,
        interval:       [0.0, 5.0], // Interval in seconds for display purposes.
        canvas:         null,
        algorithms:     {
            diff: function (ms, i) {
                if (i === 0) {
                    if (i + 1 < ms.length) {
                        return Math.abs(ms[i].strength - ms[i+1].strength);
                    } else {
                        return 0;
                    }
                } else {
                    if (i + 1 < ms.length) {
                        return Math.abs(ms[i].strength - ms[i+1].strength) + Math.abs(ms[i].strength - ms[i-1].strength);
                    } else {
                        return Math.abs(ms[i].strength - ms[i-1].strength);
                    }
                }
            },

            harmonics: function (ms, i) {
                if (ms[i].freq > 100) {
                    var f2i = find_freq(ms, ms[i].freq * 2);
                    var f3i = find_freq(ms, ms[i].freq * 3);
                    var f4i = find_freq(ms, ms[i].freq * 4);
                    var f5i = find_freq(ms, ms[i].freq * 5);
                    return add_norms(ms[i].strength, f2i.strength, f3i.strength, f4i.strength, f5i.strength);
                } else {
                    return ms[i].strength;
                }
            },

            strength: function (ms, i) { 
                return ms[i].strength; 
            },

            energy: function (ms, i) { 
                var s = ms[i].strength; 
                return s * s; 
            },

            peaks: function (ms, i) {
                if (i > 0 && i + 1 < ms.length) {
                    if (ms[i].strength > ms[i-1].strength && ms[i].strength > ms[i+1].strength) {
                        return ms[i].strength;
                    }
                } else {
                    return 0.0;
                }
            }
        },

        setup:          function (samples) {
            if (samples !== scanner.buffer) { 
                scanner.spectrogram_drawn = false; 
            }

            scanner.buffer = samples;
            if (scanner.interval[0] > scanner.buffer.duration || scanner.interval[1] > scanner.buffer.duration) {
                scanner.interval[0] = 0;
                scanner.interval[1] = scanner.buffer.duration;
                scanner.time = 0;
                elem('t1').innerHTML = declim[3](scanner.interval[0]);
                elem('t2').innerHTML = declim[3](scanner.interval[1]);
            }

            elem('now_at').innerHTML = declim[3](scanner.time);

            scanner.measure = function (freq, t) {
                var p = periodicity(samples.getChannelData(0), samples.sampleRate, scanner.window_secs, freq, t);
                var h = heterodyne(samples, t, t + scanner.window_secs * 1.5, scanner.window_secs * 0.5, 1, freq, 1);
                if (false) {
                    var pe = h.pitch_est(1);
                    p.reassigned_freq = pe ? pe.mean_hz : h.freq(0, 1);
                } else {
                    p.reassigned_freq = h.freq(0, 1);
                }
                return p;
            };

            // Keep a reference to what the measure function is measuring
            // and for what duration right in the measure function itself.
            // The callers of measure can make use of this info if necessary.
            scanner.measure.buffer = samples;
            scanner.measure.samples = samples.getChannelData(0);
            scanner.measure.window_secs = scanner.window_secs;

            scanner.measure_component = function (f) {
                return scanner.measure(f, scanner.time);
            };

            // scanner.spectrum = new Spectrum(refFreqLow_Hz, refFreqHigh_Hz, scanner.measure, scanner.time);
            debugger;
            scanner.spectrum = (new Spectrum()).setup(samples.sampleRate, Math.floor(samples.sampleRate * scanner.window_secs), refFreqLow_Hz, refFreqHigh_Hz);
            scanner.spectrum.samples = samples.getChannelData(0);
            scanner.spectrum.update(scanner.time);
            scanner.draw();
            if (!scanner.spectrogram_drawn) {
                var specCanvas = elem('spectrogram');
                var updateSpectrum = function (e) {
                    var coords = specCanvas.relMouseCoords(e);
                    var time_secs = scanner.interval[0] + (scanner.interval[1] - scanner.interval[0]) * coords.x / specCanvas.width;
                    scanner.time = time_secs;
                    scanner.setup(scanner.buffer);
                };
                //XXXX                                specCanvas.onmousemove = updateSpectrum;
                specCanvas.onclick = function (e) {
                    updateSpectrum(e);
                    if (scanner.gaussian_model) {
                        elem('gmm_coverage').innerHTML = declim[3](scanner.gaussian_model.coverage());
                    }
                    scanner.play();
                    //                                    scanner.find_peaks('peaks');
                };
                scanner.draw_spectrogram();
            }
        },

        draw:           function () {
            var ms = scanner.spectrum;
            var i, len, x, y, w, h, f = ms.toFreq_Hz;

            function plot_energy(ctxt, energy, preserve_state) {
                if (preserve_state) {
                    ctxt.save();
                }
                ctxt.fillStyle = 'black';
                ctxt.clearRect(0, 0, ctxt.canvas.width, ctxt.canvas.height);
                ctxt.fillRect(0, 0, ctxt.canvas.width, ctxt.canvas.height);

                ctxt.strokeStyle = 'yellow';
                ctxt.lineWidth = 3;
                ctxt.beginPath();
                ctxt.moveTo(0, ctxt.canvas.height);

                var i, x, y;
                for (i = 0; i < ms.length; ++i) {
                    y = Math.sqrt(energy[i]);
                    x = (ms.frequency(i) - ms.fromFreq_Hz) * ctxt.canvas.width / (ms.toFreq_Hz - ms.fromFreq_Hz);
                    ctxt.moveTo(x, ctxt.canvas.height);
                    ctxt.lineTo(x, ctxt.canvas.height * (1 - y));
                }
                ctxt.stroke();

                if (preserve_state) {
                    ctxt.restore();
                }
            };

            plot_energy(elem('spectral_scan_re').getContext('2d'), ms.reassigned_energy, true);

            var ctxt = scanner.canvas.getContext('2d');
            ctxt.save();
            plot_energy(ctxt, ms.energy, false);

            ctxt.strokeStyle = 'red';
            ctxt.lineWidth = Math.max(1, 5 * Math.log(20 * scanner.spectrum.signal) / Math.LN2);
            ctxt.beginPath();
            ctxt.moveTo(0, (1 - Math.min(0.95, 20 * scanner.spectrum.signal)) * scanner.canvas.height);
            ctxt.lineTo(scanner.canvas.width, (1 - Math.min(0.95, 20 * scanner.spectrum.signal)) * scanner.canvas.height);
            ctxt.stroke();

            // Draw the phase curve.
            if (false) {
                // Phase curve disabled for now, since I'm not
                // doing anything with that information.
                ctxt.strokeStyle = 'green';
                ctxt.lineWidth = 0.5;
                ctxt.beginPath();
                y = ms.phase(0) / Math.PI;
                ctxt.moveTo(0, 0.5 * scanner.canvas.height);
                for (i = 0; i < ms.length; ++i) {
                    y = ms.phase(i) / Math.PI;
                    x = (ms.frequency(i) - ms.frequency(0)) * scanner.canvas.width / (f - ms.frequency(0));
                    ctxt.lineTo(x, 0.5 * scanner.canvas.height * (1 - y));
                }

                ctxt.stroke();
            }

            // Draw the current gaussian mixture model if present.
            if (scanner.gaussian_model) {
                var mods = scanner.gaussian_model, m;
                scanner.gaussian_model.comps = scanner.spectrum.components;
                ctxt.strokeStyle = 'green';
                ctxt.fillStyle = 'green';
                ctxt.lineWidth = 1;
                ctxt.globalAlpha = 0.75;
                var fdx = (f - ms.frequency(0)) / ctxt.canvas.width; // Change in frequency for unit change in x coordinate.
                var peaknorm = mods.peakNorm(fdx);
                var fx;
                for (i = 0, len = mods.length; i < len; ++i) {
                    y = ctxt.canvas.height;
                    ctxt.beginPath();
                    ctxt.moveTo(0, y);
                    for (x = 0, w = ctxt.canvas.width; x < w; ++x) {
                        fx = ms.frequency(0) + x * fdx;
                        y = ctxt.canvas.height * (1 - peaknorm * mods.probability_of(i, fx - 0.5 * fdx, fx + 0.5 * fdx));
                        ctxt.lineTo(x, y);
                    }
                    ctxt.stroke();
                    ctxt.fill();
                }
            }

            ctxt.restore();

            // Draw a line indicating the time.
            var sgctxt = elem('spectrogram').getContext('2d');
            if (scanner.last_drawn_time) {
                sgctxt.putImageData(scanner.last_drawn_time.image_data, scanner.last_drawn_time.x - 2, 0); 
            }

            x = sgctxt.canvas.width * (scanner.time - scanner.interval[0]) / (scanner.interval[1] - scanner.interval[0]);
            scanner.last_drawn_time = {
                time: scanner.time,
                x: x,
                image_data: sgctxt.getImageData(x - 2, 0, 4, sgctxt.canvas.height)
            };
            sgctxt.save();
            sgctxt.strokeStyle = 'white';
            sgctxt.lineWidth = 1;
            sgctxt.globalAlpha = 0.8;
            sgctxt.beginPath();
            sgctxt.moveTo(x, 0);
            sgctxt.lineTo(x, sgctxt.canvas.height);
            sgctxt.stroke();
            sgctxt.restore();
        },

        draw_heterodyne: function (sgctxt) {
            debugger;
            var root_hz = 160;
            var harmonics = multi_heterodyne(this.buffer, this.interval[0], this.interval[1], this.window_secs, 1, frequencies(root_hz, 0, 12, 2), 10); 
            debugger;

            // Find the max energy of all the frequencies over all time in this interval.
            var max_energy_dbm = (function (i, j, N, e, hi, p) {
                e = -1e10;

                for (i = 0; i < harmonics.length; ++i) {
                    hi = harmonics[i];
                    for (j = 0, N = hi.length; j < N; ++j) {
                        p = hi.pitch_est(j);
                        e = Math.max(e, p.energy_dbm);
                    }
                }

                return e;
            })();

            // Now we can gauge the energies of individual pitch estimations relative to this
            // global maximum.

            (function (i, j, N, p, h, brk, t, x, y, dx, dy, f) {

                sgctxt.save();
                sgctxt.strokeStyle = 'pink';
                sgctxt.lineWidth = 1;
                sgctxt.globalAlpha = 0.75;

                dy = 1;

                for (j = 0; j < harmonics.length; ++j) {
                    h = harmonics[j];
                    sgctxt.beginPath();

                    for (i = 1, N = h.length, brk = true; i < N; ++i) {
                        f = h.freq(0, i);
                        p = h.pitch_est(i, 1);
                        if (p && Math.abs(12 * Math.log(f / h.freqs[0]) / Math.LN2) < 5  ) {
                            t = h.time(i);
                            x = sgctxt.canvas.width * (t - scanner.interval[0]) / (scanner.interval[1] - scanner.interval[0]);
                            dx = sgctxt.canvas.width * h.src.window_secs / (scanner.interval[1] - scanner.interval[0]);
                            y = sgctxt.canvas.height * (1 - (1 + Math.log(p.mean_hz / root_hz) / Math.LN2) / 5);
                            dy = sgctxt.canvas.height * 0.2 * p.herr_cents / 1200;
                            sgctxt.moveTo(x, y);
                            sgctxt.lineTo(x + dx, y);
                        } else {
                            brk = true;
                        }
                    }

                    sgctxt.stroke();
                }

                sgctxt.restore();
            })();
        },

        draw_selfsim_matrix: function (canv, m) {
            var N = Math.sqrt(m.length);
            canv.width = canv.height = N;
            var ctxt = canv.getContext('2d');
            var i,j,k,c,im;
            im = ctxt.createImageData(N, N);
            for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                    k = (i * N + j);
                    c = Math.min(255, Math.round(256 * m[k]));
                    im.data[k * 4 + 0] = im.data[k * 4 + 1] = im.data[k * 4 + 2] = c;
                    im.data[k * 4 + 3] = 255;
                }
            }
            ctxt.putImageData(im, 0, 0);
        },

        draw_tracked_pitches: function (ctxt, freqs, track) {
            ctxt.save();
            var f, i, len, x1, x2, y1, y2, tracks, dx, dy, ys, trkf, m, n, mN, nN, p1, p2, p12;
            tracks = track.tracks;
            ys = freqs.map(function (hz) { return 12 * Math.log(hz/refFreq_Hz) / Math.LN2; });
            var ymin = Math.min.apply(null, ys);
            var ymax = Math.max.apply(null, ys);
            var N = tracks[0].length;
            dy = ctxt.canvas.height / (ymax - ymin + 1);
            ctxt.fillStyle = 'yellow';
            if (false) {
                for (f = 0; f < freqs.length; ++f) {
                    y1 = ctxt.canvas.height - dy * (ys[f] - 0.4 - ymin);
                    y2 = ctxt.canvas.height - dy * (ys[f] + 0.4 - ymin);
                    dx = ctxt.canvas.width / tracks[f].length;
                    for (i = 0, trkf = tracks[f], len = trkf.length; i < len; ++i) {
                        x1 = i * dx;
                        x2 = (i + 1) * dx;
                        ctxt.globalAlpha = 0.95 * trkf[i];
                        ctxt.fillRect(x1, y1, x2 - x1, y2 - y1);
                    }
                }
            }

            dx = ctxt.canvas.width / N;
            ctxt.strokeStyle = 'yellow';
            ctxt.lineWidth = 2;
            ctxt.globalAlpha = 1.0;
            //            ctxt.beginPath();
            //            ctxt.moveTo(0, ctxt.canvas.height);
            var time2x = (function (t0, xscale) {
                t0 = scanner.interval[0];
                xscale = ctxt.canvas.width / (scanner.interval[1] - scanner.interval[0]);
                return function (t) { return (t - t0) * xscale; };
            })();

            var dt = scanner.step_frac * scanner.window_secs;

            var maxn, maxnval, maxm, maxmval, k, kupto, ki, ksum, ktotal, kk, n1, n2;            
            for (i = 0; i < N; ++i) {

                p1 = track.peaks[i];
                x1 = time2x(p1.time_secs);
                x2 = time2x(p1.time_secs + scanner.window_secs);

                for (m = 0, n = p1.sumOfWeights(); m < p1.length; ++m) {
                    ctxt.globalAlpha = Math.pow(p1.power, 0.15) * p1.weight(m) / n;
                    ctxt.beginPath();
                    y1 = ctxt.canvas.height * (1 - Math.log(p1.mean(m) / refFreqLow_Hz) / Math.log(refFreqHigh_Hz / refFreqLow_Hz));
                    ctxt.moveTo(x1, y1);
                    ctxt.lineTo(x2, y1);
                    ctxt.stroke();
                }

                if (i > 0) {
                    p2 = track.peaks[i - 1];
                    p12 = GMM.outer_product(p1, p2, 10);
                    for (m = 0, n = Math.pow(p1.power, 0.15), n2 = p2.length; m < p12.length; ++m) {
                        ctxt.globalAlpha = n * p12.weight(m);
                        ctxt.beginPath();
                        ctxt.moveTo(time2x(p2.time_secs + 0.8 * dt), ctxt.canvas.height * (1 - Math.log(p2.mean(m % n2) / refFreqLow_Hz) / Math.log(refFreqHigh_Hz / refFreqLow_Hz)));
                        ctxt.lineTo(x1, ctxt.canvas.height * (1 - Math.log(p1.mean(Math.floor(m / n2)) / refFreqLow_Hz) / Math.log(refFreqHigh_Hz / refFreqLow_Hz)));
                        ctxt.stroke();
                    }
                }

                if (false) {
                    ki = [];
                    for (k = 0; k < p12.length; ++k) {
                        ki.push(k);
                    }
                    ki.sort(function (i, j) { return p12.weight(j) - p12.weight(i); });
                    ktotal = 0;
                    for (k = 0; k < p12.length; ++k) {
                        ktotal += p12.weight(k); 
                    }
                    for (k = 0, ksum = 0, kupto = 0; k < p12.length; ++k) {
                        ksum += p12.weight(ki[k]);
                        if (ksum > 0.75 * ktotal) {
                            kupto = k;
                            break;
                        }
                    }

                    // Now kupto has the "upto" value.
                    for (k = 0; k <= kupto; ++k) {
                        // Draw these connections.
                        n = ki[k] % p2.length;
                        m = Math.round((ki[k] - n) / p2.length);
                        ctxt.globalAlpha = 1;
                        if (Math.abs(Math.log(p1.mean(m)/p2.mean(n))) < 3/12) {
                            ctxt.beginPath();
                            ctxt.moveTo(x1, ctxt.canvas.height * (1 - Math.log(p1.mean(m) / refFreqLow_Hz) / Math.log(refFreqHigh_Hz / refFreqLow_Hz)));
                            ctxt.lineTo(x2, ctxt.canvas.height * (1 - Math.log(p2.mean(n) / refFreqLow_Hz) / Math.log(refFreqHigh_Hz / refFreqLow_Hz)));
                            ctxt.stroke();
                        } else {
                            ctxt.beginPath();
                            ctxt.moveTo(x1, ctxt.canvas.height * (1 - Math.log(p1.mean(m) / refFreqLow_Hz) / Math.log(refFreqHigh_Hz / refFreqLow_Hz)));
                            ctxt.lineTo(x2, ctxt.canvas.height * (1 - Math.log(p1.mean(m) / refFreqLow_Hz) / Math.log(refFreqHigh_Hz / refFreqLow_Hz)));
                            ctxt.stroke();
                        }
                    }
                }
            }
            ctxt.restore();
        },

        draw_spectrogram: function (force) {
            var sgctxt = elem('spectrogram').getContext('2d');

            elem('spectrogram').waveform.draw();

            if ((!force) && scanner.interval[1] - scanner.interval[0] > 0.1) {
                // Don't draw spectrogram for too long sounds.
                return;
            }

            var sv_pitches = svaras_to_pitches(elem('svaras').value);
            var sv_freqs = sv_pitches.map(function (p) { return Math.pow(2, p/12) * refFreq_Hz; });
            var three_octaves = frequencies(refFreq_Hz, -12, 24, 1);
            var one_octave = frequencies(refFreq_Hz, 0, 12, 1);
            var chosen_frequencies = one_octave;

            debugger;
            var track = GMM.track_pitches(scanner.measure, chosen_frequencies, scanner.interval[0], scanner.interval[1], scanner.window_secs, scanner.step_frac * scanner.window_secs);
            scanner.draw_tracked_pitches(sgctxt, chosen_frequencies, track);
            debugger;
            var selfsim = GMM.self_similarity(track.peaks, 10);
            scanner.draw_selfsim_matrix(elem('selfsim'), selfsim);
            scanner.spectrogram_drawn = true;
            debugger;
            return;

            this.draw_heterodyne(sgctxt);
            //            scanner.spectrogram_drawn = true;
            //            return;

            var model__temp = find_initial_peaks_around_hz(refFreqLow_Hz, refFreqHigh_Hz, scanner.measure, scanner.interval[0]);

            var sv_filters = model__temp.spectrum.make_comb_filters(sv_freqs, 3);

            var step_secs = scanner.window_secs * 0.2;
            var track_fn = GMM.track(scanner.measure, scanner.interval[0], scanner.interval[1], scanner.window_secs, step_secs);

            // TODO: This is a stub so the rest of the algorithm can work.
            // Just collect the track results into an array for the rest of the 
            // algo.
            var track, track_i;
            while (track_i = track_fn()) {
                track = track_i;
            }


            console.log("Number of segments = " + track.length);
            console.log(JSON.stringify(track));
            debugger;

            var a, f0, i, j, k, l, p, q, t, M, N, x, y1, y2, dy, pk;
            sgctxt.save();
            sgctxt.strokeStyle = 'yellow';
            sgctxt.lineWidth = step_secs * sgctxt.canvas.width / (scanner.interval[1] - scanner.interval[0]);
            f0 = Math.log(refFreq_Hz);
            for (i = 0, M = track.length; i < M; ++i) {
                t = track[i];
                p = t.initial_peaks;
                k = 0;

                // Draw a thin half white line to indicate start of track segment.
                x = sgctxt.canvas.width * (p.time - scanner.interval[0]) / (scanner.interval[1] - scanner.interval[0]);
                sgctxt.save();
                sgctxt.globalAlpha = 0.7;
                sgctxt.lineWidth = 1;
                sgctxt.strokeStyle = 'white';
                sgctxt.beginPath();
                sgctxt.moveTo(x, 0);
                sgctxt.lineTo(x, sgctxt.canvas.height);
                sgctxt.stroke();
                sgctxt.restore();

                do {
                    x = sgctxt.canvas.width * (p.time - scanner.interval[0]) / (scanner.interval[1] - scanner.interval[0]);

                    N = p.peaks.length;
                    for (j = 0, pk = 0; j < N; ++j) {
                        q = Math.log(p.peaks.mean(j)/Math.sqrt(p.peaks.variance(j)))/Math.LN2;
                        if (q > 4) {                        
                            pk = Math.max(p.peaks.weight(j));
                        }
                    }

                    for (j = 0; j < N; ++j) {
                        a = p.peaks.weight(j)/pk;
                        q = Math.log(p.peaks.mean(j)/Math.sqrt(p.peaks.variance(j)))/Math.LN2;
                        if (q > 3) {
                            y1 = (Math.log(p.peaks.mean(j) - Math.sqrt(p.peaks.variance(j))) - f0) / Math.LN2;
                            y1 = (1 + y1) / 5;
                            y1 = sgctxt.canvas.height * (1 - y1);
                            y2 = (Math.log(p.peaks.mean(j) + Math.sqrt(p.peaks.variance(j))) - f0) / Math.LN2;
                            y2 = (1 + y2) / 5;
                            y2 = sgctxt.canvas.height * (1 - y2);
                        } else {
                            continue;
                            y1 = (Math.log(p.peaks.mean(j)) - f0) / Math.LN2;
                            y1 = (1 + y1) / 3;
                            y1 = sgctxt.canvas.height * (1 - y1);
                            y2 = y1;
                            y1 -= 2;
                            y2 += 2;
                        }
                        sgctxt.globalAlpha = a;
                        sgctxt.beginPath();
                        sgctxt.moveTo(x, y1);
                        sgctxt.lineTo(x, y2);
                        sgctxt.stroke();
                    }

                    if (k < t.peaks.length) {
                        p = t.peaks[k];
                        ++k;
                    } else {
                        break;
                    }
                } while (true);
            }
            debugger;
            sgctxt.restore();
            scanner.spectrogram_drawn = true;
        },

        play:           function () {
            var src = audio_context.createBufferSource();
            src.buffer = this.buffer;
            src.gain.value = 0.5;
            src.connect(audio_context.destination);
            src.noteGrainOn(0, this.time, Math.min(2.0, scanner.interval[1] - this.time));//this.buffer.duration - this.time);
        },
        find_peaks:     function (result_id) {
            var result = elem(result_id);
            var peaks = find_peaks_around(refFreq_Hz, -12, 24, scanner.measure, scanner.time);
            var peaks2 = declim_gaussian_models(peaks.peaks);
            result.innerHTML = JSON.stringify(peaks2);
        }
    };

    var alerter = function (msg) {
        return function (e) { alert(msg + e); };
    };

    function process_file(file, callback) {
        var reader = new FileReader();
        reader.onabort = alerter('File read aborted!');
        reader.onerror = alerter('File read error!');
        reader.onload = function () {
            try {
                var buffer = audio_context.createBuffer(reader.result, true);
                Waveform.setup(elem('spectrogram'), buffer);
                callback(buffer);
            } catch (e) {
                enable = false;
                reader.onerror(); 
            }
        };
        reader.readAsArrayBuffer(file);
    }

    exports.configure_file_picker = function (input_id) {
        var input = elem(input_id);
        input.setAttribute('accept', 'audio/mpeg, audio/wav');
        input.addEventListener('change', function () {
            process_file(input.files[0], scanner.setup);
        });
    };

    exports.configure_spectral_scanner = function (display_id) {
        var canvas = elem(display_id);
        scanner.canvas = canvas;

        elem('plot_spectrogram').addEventListener('click', function () {
            scanner.draw_spectrogram();
        });

        function update_interval(t1, t2) {
            scanner.last_drawn_time = null;

            if (scanner.buffer) {
                var t1a = t1, t2a = t2;
                t1a = Math.max(0, t1);
                if (t1a > t1) { t2a = Math.min(scanner.buffer.duration, t1a + (t2 - t1)); }
                if (t2a < t2) { t1a = Math.max(0, t2a - (t2 - t1)); }
                t1 = t1a;
                t2 = t2a;
            }

            scanner.interval[0] = t1;
            scanner.interval[1] = t2;

            if (scanner.buffer) {
                elem('spectrogram').waveform.interval = scanner.interval;
            }

            elem('t1').innerHTML = ''+declim[2](t1);
            elem('t2').innerHTML = ''+declim[2](t2);

            if (scanner.buffer) {
                scanner.draw_spectrogram();
            }
        }

        update_interval(scanner.interval[0], scanner.interval[1]);

        elem('tw_left').onclick = function () {
            var t1 = scanner.interval[0];
            var t2 = scanner.interval[1];

            // Shift left by 30% and keep the interval duration the same.
            var dt = (t2 - t1);
            t1 = Math.max(0.0, t1 - dt * 0.3);
            t2 = Math.min(scanner.buffer.duration, t1 + dt);

            update_interval(t1, t2);
        };

        elem('tw_right').onclick = function () {
            var t1 = scanner.interval[0];
            var t2 = scanner.interval[1];

            // Shift right by 30% and keep the interval duration the same.
            var dt = t2 - t1;
            t2 = Math.min(scanner.buffer.duration, t2 + dt * 0.3);
            t1 = Math.max(0, t2 - dt);

            update_interval(t1, t2);
        };

        elem('tw_zoomin').onclick = function () {
            var t1 = scanner.interval[0];
            var t2 = scanner.interval[1];
            var t = scanner.time || ((t1 + t2) / 2);
            var dt = Math.max(0.1, (t2 - t1) * 1 / 6);

            if (t2 - t1 >= 6) {
                dt = Math.max(2.999, dt);
            }

            // Zoom in to 2/3 of the original width.
            t1 = t - dt;
            t2 = t + dt;

            update_interval(t1, t2);
        };

        elem('tw_zoomout').onclick = function () {
            var t1 = scanner.interval[0];
            var t2 = scanner.interval[1];

            // Zoomout so that a zoomin that follows will
            // return the interval to the same state as far
            // as possible.
            var t = (t1 + t2) / 2;
            var dt = (t - t1) * 3;

            if (t - t1 < 3) {
                dt = Math.min(3.001, dt);
            }

            t1 = Math.max(0, t1 - dt);
            t2 = Math.min(scanner.buffer.duration, t2 + dt);

            update_interval(t1, t2);
        };

        elem('track_window').onclick = function () {
            scanner.draw_spectrogram(true);
        };

        function initialize_gaussian_model(scanner) {
            scanner.gaussian_model = find_initial_peaks_around_hz(refFreqLow_Hz, refFreqHigh_Hz, scanner.measure, scanner.time);
        }

        function show_gaussian_coverage(scanner) {
            var coverage;
            coverage = scanner.gaussian_model.coverage();
            elem('gmm_coverage').innerHTML = declim[3](coverage);
        }

        function display_gaussian_model(scanner) {
            elem('peaks').innerHTML = ('('+declim[3](scanner.gaussian_model.sumOfWeights())+')') + scanner.gaussian_model.toString(+elem('tonic').value);
            scanner.draw();
        }

        elem('gmm_begin').onclick = function () {
            initialize_gaussian_model(scanner);
            display_gaussian_model(scanner);
        };

        elem('gmm_step').onclick = function () {
            if (!scanner.gaussian_model) {
                initialize_gaussian_model(scanner);
            }
            show_gaussian_coverage(scanner);
            var g = scanner.gaussian_model.iterate(scanner.gaussian_model.spectrum, 1);
            var stability = GMM.convergence(scanner.gaussian_model, g);
            scanner.gaussian_model = g;
            elem('gmm_stability').innerHTML = declim[3](stability);
            display_gaussian_model(scanner);
        };

        var track_peaks = false;

        elem('gmm_stabilize').onclick = function () {
            if (!scanner.gaussian_model) {
                initialize_gaussian_model(scanner);
            }
            show_gaussian_coverage(scanner);

            var p = GMM.stabilize(scanner.spectrum, scanner.gaussian_model, {track_peaks: track_peaks});
            var probs;
            if (scanner.gaussian_model.length === p.length) {
                probs = GMM.tracking_probabilities(scanner.gaussian_model, p);//.map(declim[5]);
            } else {
                probs = "model simplified";
            }
            elem('jump_probabilities').innerHTML = JSON.stringify(probs);
            scanner.gaussian_model = p;

            //            elem('gmm_stability').innerHTML = "stable";
            display_gaussian_model(scanner); 
        };

        elem('gmm_track').onclick = function () {
            track_peaks = true;
            elem('gmm_stabilize').onclick();
            track_peaks = false;
            show_gaussian_coverage(scanner);
        };

        elem('gmm_simplify').onclick = function () {
            alert('disabled!');
            return;

            if (!scanner.gaussian_model) {
                initialize_gaussian_model(scanner);
            }

            var p = scanner.gaussian_model.peaks.slice(0); // Make a copy.
            if (p.length > 4) {
                p.sort(function (a, b) { return b[0] - a[0]; });
                // Keep half of the peaks.
                var i, len, sum = 0;
                for (i = 0, len = p.length; i < len; ++i) {
                    sum += p[i][0];
                }
                var thresh = sum * 0.01;
                for (i = 0, len = p.length; i < len; ++i) {
                    if (p[i][0] < thresh) {
                        p.splice(i);
                        break;
                    }
                }
                scanner.gaussian_model.peaks = p;
                display_gaussian_model(scanner);
            }
        };

        elem('gmm_simplify_more').onclick = function () {
            alert('disabled!');
            return;

            if (!scanner.gaussian_model) {
                initialize_gaussian_model(scanner);
            }

            var p = scanner.gaussian_model.peaks.slice(0); // Make a copy.
            var i, len, thresh;
            p.sort(function (a, b) { return b[0] - a[0]; });

            thresh = p[0][0] * 0.25;
            for (i = 0, len = p.length; i < len; ++i) {
                if (p[i][0] < thresh) {
                    p.splice(i);
                    break;
                }
            }
            scanner.gaussian_model.peaks = p;
            display_gaussian_model(scanner);
        };
    };

    return exports;
})({}, Math, Date);
