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
            "backpropagation": function (net, newInputData, validOutputData, gradientFn) {
                var bp = this;
                bp.done = function () {
                };

                function getErrWrtTotalNetInput(out, targetOut) {
                    return -(targetOut - out) * out * (1 - out);
                }



                var result, i, neuronN,
                    currentL, neuron, gradient = 0, outputLayer = true;

                for (i = net.layer.length; i >= 0; i--) {
                    currentL = net.layer[i];
                    for (neuronN in currentL) {
                        neuron = currentL[neuronN];

                        if (outputLayer) {
                            neuron.errorsTotalNetInput = getErrWrtTotalNetInput(neuron.signal, validOutputData[neuronN]);
                        }

                        if (!outputlayer && i > 0) { //must be a hidden layer then
                            var laterLayer = net.layer[i + 1];
                            var errorsWrtOut = 0;    
                            for (var neuronL in laterLayer) {
                                var laterNeuron = laterLayer[neuronL];
                                errorsWrtOut += laterNeuron.errorsTotalNetInput * neuron[neuronN].weight[neuronL];
                            }

                            neuron.errHiddenToTotalNetInput = errorsWrtOut * neuron.signal * (1 - neuron.signal);
                        }


                        if (i === 0) { //input layer

                        }


                        neuron.weight *= gradient;

                    }
                    outputLayer = false;
                }

                //bp.done(data);

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
            "sigmoid": function (connections) {
                var e = 1, ConNeuron, conLength = connections.length, nsignal;
                for (var current in connections) {
                    ConNeuron = connections[current];

                    this.inputs.push(ConNeuron); //save all the input connections/neurons that relate to this neuron

                    if (conLength < current) {
                        nsignal = connections[current + 1];
                    } else {
                        nsignal = { weight: 0, signal: 0 } // could add bias node here ...
                    }
                    e = e + ConNeuron.weight * ConNeuron.signal + nsignal.weight * nsignal.signal;
                }
                this.signal = 1 / (1 + Math.pow(Math.E, -e)); // where Math.E is 2.718281828459045, why do we calculate with 2.7...?

                return this; //this is the current neuron
            }
        };
        /**
         *
         */
        me.growNeuron = function (weight, threshold, activationMethod) {
            return {
                weight: [weight],
                threshold: threshold,
                activate: me.activationStyles[activationMethod]
            };
        };

        return me;
    } ();

    /**
     *
     */
    corpus.growNeuronsRandom = function (amount) {
        var randomElements = [];
        for (var i = 0; i < amount - 1; i++) {
            randomElements.push({ weight: Math.random(), threshold: 0 });
        }

        randomElements.push({ weight: Math.random(), threshold: 1 }); //add a bias neuron

        return corpus.growNeurons(randomElements, 'sigmoid'); // could i use a randomly selected activation function?
    }

    /**
     *
     */
    corpus.growNeurons = function (arrWT, activationType) {
        return arrWT.map(function (element) {
            return corpus.cellBuilder(element.weight, element.threshold, activationType);
        });
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
            var level = { name: name, neurons: neurons };
            net.layer.push(level);
            return net;
        };

        /**
         *
         */
        net.stimulus = function (inputSignals, activationMethod) {
            var result, i, neuronI, neuronN, activationFN =
                corpus.cellBuilder.activationStyles[activationMethod],
                currentL, nextL, connections = [];

            for (i in net.layer) {
                currentL = net.layer[i];
                if (typeof net.layer[i + 1] !== "undefined") {
                    nextL = net.layer[i + 1];
                    for (neuronN in nextL) {
                        for (neuronI in currentL) {
                            connections.push(currentL[neuronI]);
                        }
                        nextL[neuronN] = activationFN.call(nextL[neuronN], connections);
                        connections = [];
                    }
                } else {
                    result = currentL;
                }
            }

            return result;
        };

        /**
         *
         */
        net.learn = function (newInputData, validOutputData, learningStyle, gradientFn) {
            var learnFn = corpus.cellBuilder.learningStyles[learningStyle];
            return learnFn(net, newInputData, validOutputData, gradientFn);
        };

        return net;
    };


    return corpus;
};


//testing the magic//////////////////////////////////////////////////////////

var nnetFactory = new corpusCere(),
    learningOptions = { rate: 0.001, iterations: 10000, momentum: 0.2 };

var net = nnetFactory.net("testNet")
    .addLayer("inputLayer", nnetFactory.growNeuronsRandom(3))
    .addLayer("hiddenLayer", nnetFactory.growNeuronsRandom(3))
    .addLayer("outputLayer", nnetFactory.growNeuronsRandom(1));

net.learn(newInputData, validOutputData, "backpropagation", learningOptions)
    .done = function (learningResult) {
        console.log(learningResult);
    }

//after training, try some new unknown input

var result = net.stimulus(newInputData);

console.log(result);

