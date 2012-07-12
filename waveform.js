
// Module to draw audio waveforms efficiently.
var Waveform = (function (exports, global, Math) {

    var nextFrame = (window.requestAnimationFrame 
                        || window.webkitRequestAnimationFrame 
                        || window.mozRequestAnimationFrame
                        || (function (f) { return setTimeout(f, 1000/60); }));

    // Sets up the given audio buffer to be drawn
    // into the given canvas object. Adds a property
    // to the canvas object called 'waveform' which
    // is an object with methods you can call to 
    // update the waveform upon user input.
    //
    // canvas.waveform.interval
    //  A two-element array giving the time interval
    //  that is being displayed in the canvas.
    //
    //  Gettable and settable. When you set it, the
    //  canvas automatically redraws itself if
    //  necessary.
    //
    // canvas.waveform.draw()
    //  Call explicitly to redraw the waveform.
    //
    // canvas.waveform.onDraw
    //  Set to a callback that'll then get called whenever
    //  the canvas waveform refreshes.
    //
    // style.animDur_ms         = duration of waveform interval change animation.
    // style.backgroundColor    = color of drawn waveform's background.
    // style.color              = waveform's color
    // style.alpha              = waveform's opacity
    // style.lineWidth          = waveform's outline thickness
    function setup(canvas, audioBuffer, style) {
        var ctxt            = canvas.getContext('2d');
        var chMinMaxs       = channelMinMaxCalculators(audioBuffer);
        var fromTime_secs   = 0;
        var toTime_secs     = audioBuffer.duration;
        var fromIx          = 0;
        var toIx            = secstosmps(audioBuffer.duration);
        var animDur_ms      = (style && style.animDur_ms) || 250;
        var animating       = false;

        function secstosmps(secs) {
            return Math.floor(audioBuffer.sampleRate * secs);
        }

        var wave = {
            get interval() {
                return [fromTime_secs, toTime_secs];
            },

            set interval(dt) {
                var oldFrom_secs    = fromTime_secs;
                var oldTo_secs      = toTime_secs;

                fromTime_secs       = Math.max(0, Math.min(dt[0], audioBuffer.duration));
                toTime_secs         = Math.max(fromTime_secs, Math.min(dt[1], audioBuffer.duration));

                var _fromIx         = secstosmps(fromTime_secs);
                var _toIx           = secstosmps(toTime_secs);

                var animStart_ms, animNow_ms, animFrom_secs, animTo_secs;

                if (fromIx !== _fromIx || toIx !== _toIx) {
                    fromIx  = _fromIx;
                    toIx    = _toIx;
                    if (!animating) {
                        if (animDur_ms > 0) {
                            animating       = true;
                            animStart_ms    = Date.now();
                            animNow_ms      = animStart_ms;
                            animFrom_secs   = oldFrom_secs;
                            animTo_secs     = oldTo_secs;
                            nextFrame(function () {
                                var frac = Math.max(0, Math.min((Date.now() - animStart_ms) / animDur_ms, 1));
                                draw(secstosmps(oldFrom_secs + frac * (fromTime_secs - oldFrom_secs)), 
                                    secstosmps(oldTo_secs + frac * (toTime_secs - oldTo_secs)));
                                if (frac < 1) {
                                    nextFrame(arguments.callee);
                                } else {
                                    animating = false;
                                }
                            });
                        } else {
                            draw(fromIx, toIx);
                        }
                    }
                }
            }
        };

        function draw(fromIx, toIx) {
            if (toIx <= fromIx) {
                return;
            }

            ctxt.save();

            ctxt.clearRect(0, 0, canvas.width, canvas.height);
            ctxt.fillStyle = (style && style.backgroundColor) || 'black';
            ctxt.fillRect(0, 0, canvas.width, canvas.height);

            var i1                  = fromIx;
            var i2                  = toIx;
            var samp_per_pix        = (i2 - i1) / canvas.width;
            var dsamp               = 1 / samp_per_pix;
            var i, N, low = 1, high = -1, pix = 0, dpix = 0, x, y;
            var samples             = audioBuffer.getChannelData(0);
            var minsample           = chMinMaxs[0].min;
            var maxsample           = chMinMaxs[0].max;

            ctxt.strokeStyle        = (style && style.color) || 'red';
            ctxt.fillStyle          = (style && style.color) || 'red';
            ctxt.globalAlpha        = (style && style.alpha) || 0.4;
            ctxt.lineWidth          = (style && style.lineWidth) || 1;

            ctxt.beginPath();
            ctxt.moveTo(0, 0.5 * canvas.height);
            for (i = 0, N = canvas.width; i < N; ++i) {
                high = maxsample(Math.floor(i1 + i * samp_per_pix), Math.floor(i1 + (i + 1) * samp_per_pix));
                ctxt.lineTo(i, 0.5 * canvas.height * (1.0 - high));
            }
            for (i = canvas.width - 1; i >= 0; --i) {
                low = minsample(Math.floor(i1 + i * samp_per_pix), Math.floor(i1 + (i + 1) * samp_per_pix));
                ctxt.lineTo(i, 0.5 * canvas.height * (1.0 - low));
            }
            ctxt.stroke();
            ctxt.fill();

            ctxt.restore();

            // Call the refresh callback if specified.
            if (wave.onDraw) {
                wave.onDraw(canvas);
            }
        };

        wave.draw = function () { draw(fromIx, toIx); };
        canvas.waveform = wave;
        return canvas;        
    }


    // Returns an array of minmax calculators for
    // each channel in the audio buffer.
    function channelMinMaxCalculators(audioBuffer) {
        var i, N, mm = [];
        for (i = 0, N = audioBuffer.numberOfChannels; i < N; ++i) {
            mm.push(minMaxCalculator(audioBuffer.getChannelData(i)));
        }
        return mm;
    }

    // Returns a function of index range for calculating the minmax
    // values of a waveform within any range. Progressively caches
    // the samples mipmaps for speed purposes.
    function minMaxCalculator(samples) {
        var mipmapLo = [samples], mipmapHi = [samples];
        var starting_pof2 = 256;
        var starting_pof2ix = 8;

        // Create the mipmaps.
        var i, j, k, N, len, l, h, pl, ph;
        for (i = 1; i <= starting_pof2ix; ++i) {
            pl = mipmapLo[i-1];
            ph = mipmapHi[i-1];
            len = pl.length;
            N = Math.floor((len + 1) / 2);
            mipmapLo[i] = l = new Float32Array(N);
            mipmapHi[i] = h = new Float32Array(N);
            for (j = 0, k = 0; k < len; ++j, k += 2) {
                l[j] = Math.min(pl[k], pl[k + 1]);
                h[j] = Math.max(ph[k], ph[k + 1]);
            }
        }

        function max(fromIx, toIx, pof2ix, pof2ixval) {
            if (toIx <= fromIx) { return samples[fromIx]; }
            var i, len, N, l, j, hi = -1, ix = pof2ix;
            if (pof2ix === 0) {
                for (i = fromIx, N = Math.min(toIx, samples.length); i < N; ++i) {
                    hi = Math.max(hi, samples[i]);
                }
                return hi;
            }
            var loIx = Math.floor((fromIx + pof2ixval - 1) / pof2ixval);
            var hiIx = Math.floor(toIx / pof2ixval);
            if (hiIx - loIx > 0) {
                for (i = loIx; i < hiIx; ++i) {
                    hi = Math.max(hi, mipmapHi[ix][i]);
                }

                if (pof2ix > 0) {
                    return Math.max(
                            max(fromIx, loIx * pof2ixval, pof2ix-1, pof2ixval/2), 
                            hi, 
                            max(hiIx * pof2ixval, toIx, pof2ix-1, pof2ixval/2));
                } else {
                    return hi;
                }
            } else {
                return max(fromIx, toIx, pof2ix-1, pof2ixval/2);
            }
        }

        function min(fromIx, toIx, pof2ix, pof2ixval) {
            if (toIx <= fromIx) { return samples[fromIx]; }
            var i, len, N, l, j, lo = 1, ix = pof2ix;
            if (pof2ix === 0) {
                for (i = fromIx, N = Math.min(toIx, samples.length); i < N; ++i) {
                    lo = Math.min(lo, samples[i]);
                }
                return lo;
            }
            var loIx = Math.floor((fromIx + pof2ixval - 1) / pof2ixval);
            var hiIx = Math.floor(toIx / pof2ixval);
            if (hiIx - loIx > 0) {
                for (i = loIx; i < hiIx; ++i) {
                    lo = Math.min(lo, mipmapLo[ix][i]);
                }

                if (pof2ix > 0) {
                    return Math.min(
                            min(fromIx, loIx * pof2ixval, pof2ix-1, pof2ixval/2), 
                            lo, 
                            min(hiIx * pof2ixval, toIx, pof2ix-1, pof2ixval/2));
                } else {
                    return lo;
                }
            } else {
                return min(fromIx, toIx, pof2ix-1, pof2ixval/2);
            }
        }

        return {
            min: (function (fromIx, toIx) { return min(fromIx, toIx, starting_pof2ix, starting_pof2); }),
            max: (function (fromIx, toIx) { return max(fromIx, toIx, starting_pof2ix, starting_pof2); })
        };
    }

    exports.setup = setup;
    return exports;

}({}, window, window.Math));
