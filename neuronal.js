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
            "backpropagation": function (net, newInputData, validOutputData) {
                var bp = this;
                bp.done = function () { };

                 //next build backpropagation





                // bp.done(data);

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
                var e = 1, signal, conLength = connections.length;
                for (var current in connections) {
                    signal = connections[current];

                    if (conLength < current) {
                        nsignal = connections[current + 1];
                    } else {
                        nsignal = { weight: 0, strength: 0 } // could add bias node here ...
                    }

                    e = e + signal.weight * signal.strength + nsignal.weight * nsignal.strength;

                }
                return 1 / (1 + Math.pow(Math.E, -e)); // where Math.E is 2.718281828459045, why do we calculate with 2.7...?
            }
        };
        /**
         * 
         */
        me.growNeuron = function (weight, threshold, activationMethod) {
            return {
                weight: weight,
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
            var current, i, next, neuronI, neuronN, activationFN =
                corpus.cellBuilder.activationStyles[activationMethod],
                curNeuron, nextNeuron, connections = [];

            for (i in net.layer) {
                currentL = net.layer[i];

                if (typeof net.layer[i + 1] !== "undefined") {
                    nextL = net.layer[i + 1];

                    for (neuronN in nextL) {
                        nextNeuron = nextL[neuronN];

                        for (neuronI in currentL) {
                            curNeuron = currentL[neuronI];
                            connections.push(curNeuron);
                        }
                        nextNeuron.strength = activationFN(connections);
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
        net.learn = function (newInputData, validOutputData, learningStyle) {
            var learnFn = corpus.cellBuilder.learningStyles[learningStyle];
            return learnFn(net, newInputData, validOutputData);
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

