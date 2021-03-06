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
            "backpropagation": function (net, newInputData, validOutputData, learningOptions, iteration) {



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
                    return 0.5 * Math.pow(targetOut - out, 2);
                }


                function getErrorPerLayer(layer, targetOut) {
                    var errorTotal = 0;
                    for (var neuronN in layer.neurons) {
                        errorTotal += getError(layer.neurons[neuronN].output, targetOut[neuronN]);
                    }
                    return errorTotal;
                }

                function getResults(layer) {
                    return layer.neurons.map(function (v) {
                        return v.output;
                    });
                }


                var result, i, neuronN,
                    currentL, neuron, outputlayer = true, outputValidperLayer = validOutputData, pd_error_wrt_weight;

                // ausgabeNeuronen quotienten berechnen mit ausgabeerwartung
                // output layer pd error
                var outPutLayer = net.layer.slice(-1)[0];

                for (neuronN in outPutLayer.neurons) {
                    var neuron2 = outPutLayer.neurons[neuronN];
                    neuron2.gradient = getErrWrtTotalNetInput(neuron2.output, validOutputData[neuronN]);
                }


                // hidden layer pd error
                for (neuronN in net.layer[1].neurons) {
                    var neuron1 = net.layer[1].neurons[neuronN];
                    var erHiddenN = 0;
                    for (var h in outPutLayer.neurons) {
                        var outPutNeuron = outPutLayer.neurons[h];
                        erHiddenN += outPutNeuron.gradient * outPutNeuron.input[neuronN].weight;
                    }

                    neuron1.gradient = erHiddenN * (neuron1.output * (1 - neuron1.output));
                }



                for (i = 0; i < net.layer.length; i++) {
                    var neurons = net.layer[i].neurons;

                    for (neuronN in neurons) {
                        var neuron = neurons[neuronN];

                        if (typeof neuron.gradient === 'undefined') neuron.gradient = 0;

                        neuron.threshold -= learningOptions.LEARNINGRATE * neuron.gradient;

                        for (var inX in neuron.input) {
                            var inputConnection = neuron.input[inX];

                            if (typeof inputConnection.previousDeltas === 'undefined') inputConnection.previousDeltas = 0;

                            inputConnection.delta = learningOptions.LEARNINGRATE * neuron.gradient * inputConnection.signal;
                            inputConnection.weight += inputConnection.delta;

                        }
                      
                    }
                }


                net.doneIteration.call(net, { it: iteration, in: newInputData, out: validOutputData, result: getResults(net.layer.slice(-1)[0]), err: getErrorPerLayer(net.layer.slice(-1)[0], validOutputData) });
                return net;
            },
            "geneticAlgorithm": function () {

            }
        };

        /**
         *
         */
        me.activationStyles = {
            "input": function () {
                this.output = this.input[0].signal;
                return this;
            },
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

                this.output = 1 / (1 + Math.exp(-sum)); // where Math.E is 2.718281828459045, why do we calculate with 2.7...?

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
    corpus.growNeuronsRandom = function (amount, activation, bias) {

        var neurons = Array.apply(null, Array(amount)).map(function () {
            return corpus.cellBuilder.growNeuron(bias || 0, activation || 'sigmoid');
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
        net.done = function (val) {
            // console.log(val);
        };
        net.doneIteration = function (val, params) {
            console.log("iteration: " + params.it + " errRate: " + params.err);
        };

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
        net.stimulus = function (inputSignals) {
            var i, e, neuronN,
                currentL, nextL, allNextLNeurons;

            for (e in net.layer[0].neurons) {
                net.layer[0].neurons[e].input[0] = {
                    signal: inputSignals[e],
                    weight: 1
                };
            }

            for (i = 0; i < net.layer.length; i++) {
                currentL = net.layer[i];

                for (neuronN in currentL.neurons) {

                    currentL.neurons[neuronN].activate.call(currentL.neurons[neuronN]);

                    if (typeof net.layer[i + 1] !== 'undefined') {
                        nextL = net.layer[i + 1];
                        for (allNextLNeurons in nextL.neurons) {
                            nextL.neurons[allNextLNeurons].input[neuronN].signal = currentL.neurons[neuronN].output;
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

            for (var itera = 0; itera < learningOptions.ITERATIONS; itera++) {
                var set = Math.floor(Math.random() * newInputData.length);
                learnFn(net, newInputData[set], validOutputData[set], learningOptions, itera);
            }
            net.done.call(net, { in: newInputData, out: validOutputData, err: net.layer.slice(-1)[0].gradient });
        };

        net.persist = function() {
            return JSON.stringify(net.layer);
        }

        return net;
    };


    return corpus;





};


//testing the magic//////////////////////////////////////////////////////////

var nnetFactory = new corpusCere(),
    learningOptions = { LEARNINGRATE: .7, ITERATIONS: 70003, MOMENTUM: .7 };


var trainingset = [
    [.9, .9, .9, .9],
    [.9, .1, .9, .1],
    [.8, .9, .8, .9]
    //  [1, 1]
];

var resultset = [
    [.9],
    [.1],
    [.9]


    //  [0]
];


var net = nnetFactory.net("testNet")
    .addLayer("inputLayer", nnetFactory.growNeuronsRandom(2, "sigmoid", 0))
    .addLayer("hiddenLayer", nnetFactory.growNeuronsRandom(2, "sigmoid", 1))
    .addLayer("outputLayer", nnetFactory.growNeuronsRandom(1, "sigmoid", 1));

net.stimulus([1, 0]);

net.doneIteration = function (params) {
    this.stimulus(params.in);
    //  console.log("iteration: " + params.it, params.in, params.out, params.result, " errRate: " + params.err);

};

net.learn(trainingset, resultset, "backpropagation", learningOptions);

/*
Ergebnis:  [ 0.9, 0.9 ] 0.999999752724947
Ergebnis:  [ 0.9, 0.1 ] 0.1044124840580854
Ergebnis:  [ 0.9, 0.7 ] 0.9999928211759743
*/


for (var s in trainingset) {
    var result = net.stimulus(trainingset[s]);

    console.log("Ergebnis: ", trainingset[s], JSON.stringify(result.neurons[0].output));
}

console.log(net.persist());
