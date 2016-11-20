/**
 * @author Alexander Kindziora
 *
 */

// building my own artificial neural network using sigmoid neurons and multiple layers
// just to get a deeper understanding of ai

var corpusCere = function () {
    var corpus = this;

    /**
     *
     */
    corpus.cellBuilder = function () {
        var me = this;

        me.learningStyles = {
            "backpropagation": function (net, newInputData, validOutputData, learningOptions) {
                var bp = this;
                bp.done = function () {
                };

                var errorTotal = 0;

                function getErrWrtTotalNetInput(out, targetOut) {
                    return -(targetOut - out) * (out * (1 - out));
                }
                /**
                 * get error rate, calculate the deviation
                 * getError(2,3) = 0.5
                 * getError(2,2) = 0
                 * getError(8,2) = 18
                 * 
                 */
                function getError(out, targetOut) {
                    return Math.pow(targetOut - out, 2) / 2;
                }


                function getErrorPerLayer(layer, targetOut) {
                    var errorTotal = 0;
                    for (var neuronN in layer) {
                        errorTotal += getError(layer[neuronN].signal, targetOut);
                    }
                    return errorTotal;
                }

                var result, i, neuronN,
                    currentL, neuron, gradient = 0, outputlayer = true, outputValidperLayer = validOutputData;

                for (i = net.layer.length - 1; i >= 0; i--) { // begin at the end
                    currentL = net.layer[i];

                    currentL.errorRate = getErrorPerLayer(currentL.neurons, outputValidperLayer);

                    for (neuronN in currentL.neurons) {
                        neuron = currentL.neurons[neuronN];

                        if (outputlayer) {
                            neuron.errorRate = getErrWrtTotalNetInput(neuron.signal, validOutputData[neuronN]);
                        }

                        if (!outputlayer && i > 0) { //must be a hidden layer then
                            var laterLayer = net.layer[i + 1].neurons,
                                laterNeuron = laterLayer[neuronN];
                            erHiddenN = 0;
                            for (var h in laterLayer) {
                                erHiddenN += laterLayer[neuronN].errorRate * neuron.weights[h];
                            }

                            neuron.pderrors = erHiddenN * (neuron.signal * (1 - neuron.signal));

                            for (var h in laterLayer) {

                                var errorsWrtWeight = laterNeuron.errorRate * laterNeuron.input[h].signal;

                                laterNeuron.input[neuronN].weight -= learningOptions.LEARNINGRATE * errorsWrtWeight;

                            }

                        }

                    }
                    //for deeper nets we would need to pass new estimated optimal output
                    // here we might run into the vanishing gradient problem?
                    outputValidperLayer = [123];  // calculate new closer to valid value and set it for the next iteration
                    outputLayer = false;
                }

                bp.done(net);

                return bp;
            },
            "geneticAlgorithm": function () {

            }
        };

        /**
         *
         */
        me.activationStyles = {
            /**
             * a activation function using sigmoid
             */
            "sigmoid": function () {
                var sum = 0, weight, signal;

                for (var current in this.input) {

                    weight = this.input[current].weight;
                    signal = this.input[current].signal;

                    sum += weight * signal;
                }

                sum += this.threshold; // apply bias

                this.signal = 1 / (1 + Math.pow(Math.E, -sum)); // where Math.E is 2.718281828459045, why do we calculate with 2.7...?

                return this; //this is the current neuron
            }
        };
        /**
         *
         */
        me.growNeuron = function (threshold, activationMethod) {
            return {
                input: [{}], //input weights and signals for training reasons
                threshold: threshold,
                activate: me.activationStyles[activationMethod]
            };
        };

        return me;
    } ();

    /**
     *
     */
    corpus.growConnections = function (neuron, amount) {
        neuron.input = Array.apply(null, Array(amount)).map(function () { return { weight: Math.random() }; });
        return neuron;
    }

    /**
     * could i use a randomly selected activation function? for every layer?
     */
    corpus.growNeuronsRandom = function (amount, activation) {

        if (typeof activation === 'undefined') activation = 'sigmoid';

        var neurons = Array.apply(null, Array(amount)).map(function () {
            return corpus.cellBuilder.growNeuron(1, activation);
        });

        //  neurons.push(corpus.cellBuilder.growNeuron(1, activation)); //add a bias neuron

        return neurons;
    }



    /**
     *
     */
    corpus.net = function (name) {
        var net = this;
        net.layer = [];
        net.name = name;

        /**
         *
         */
        net.addLayer = function (name, neurons) {
            //create weight connections to previous layer
            var connectionsCnt = net.layer.length > 0 ? net.layer.slice(-1)[0].neurons.length : 0;

            for (var current in neurons) {
                corpus.growConnections(neurons[current], connectionsCnt);
            }

            var level = { name: name, neurons: neurons };
            net.layer.push(level);
            return net;
        };

        /**
         *
         */
        net.stimulus = function (inputSignals, activationMethod) {
            var i, e, neuronN, activationFN =
                corpus.cellBuilder.activationStyles[activationMethod],
                currentL, nextL, allNextLNeurons;

            for (e in net.layer[0].neurons) {
                net.layer[0].neurons[e].input[e] = {
                    signal: inputSignals[e],
                    weight: 0
                };
            }

            for (i = 0; i < net.layer.length; i++) {
                currentL = net.layer[i];

                for (neuronN in currentL.neurons) {

                    currentL.neurons[neuronN] = activationFN.call(currentL.neurons[neuronN]);

                    if (typeof net.layer[i + 1] !== 'undefined') {
                        nextL = net.layer[i + 1];
                        for (allNextLNeurons in nextL.neurons) {
                            console.log(currentL.neurons[neuronN], neuronN);

                            nextL.neurons[allNextLNeurons].input[neuronN].signal = currentL.neurons[neuronN].signal;
                        }
                    }

                }

            }

            return currentL; //output layer will be the last one
        };

        /**
         *
         */
        net.learn = function (newInputData, validOutputData, learningStyle, learningOptions) {

            var learnFn = corpus.cellBuilder.learningStyles[learningStyle];

            return learnFn(net, newInputData, validOutputData, learningOptions);
        };

        return net;
    };


    return corpus;
};


//testing the magic//////////////////////////////////////////////////////////

var nnetFactory = new corpusCere(),
    learningOptions = { LEARNINGRATE: 0.01, iterations: 100, momentum: 0.2 };

var newInputData = [1, 0];
var validOutputData = [1];

var net = nnetFactory.net("testNet")
    .addLayer("inputLayer", nnetFactory.growNeuronsRandom(2))
    .addLayer("hiddenLayer", nnetFactory.growNeuronsRandom(2))
    .addLayer("outputLayer", nnetFactory.growNeuronsRandom(1));



//before training, try some input

var result = net.stimulus(newInputData, "sigmoid");

console.log(result);

net.learn(newInputData, validOutputData, "backpropagation", learningOptions)
    .done = function (net) {
        console.log(net);
    }

//after training, try some input

var result = net.stimulus(newInputData, "sigmoid");

console.log(result);