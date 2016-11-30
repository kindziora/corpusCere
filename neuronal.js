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
                        errorTotal += getError(layer.neurons[neuronN].signal, targetOut[neuronN]);
                    }
                    return errorTotal;
                }

                function getResults(layer) {
                    return layer.neurons.map(function (v) {
                        return v.signal;
                    });
                }


                var result, i, neuronN,
                    currentL, neuron, outputlayer = true, outputValidperLayer = validOutputData, pd_error_wrt_weight;

                // ausgabeNeuronen quotienten berechnen mit ausgabeerwartung
                // output layer pd error
                for (neuronN in net.layer[2].neurons) {
                    var neuron = net.layer[2].neurons[neuronN];
                    neuron.errorRate = getErrWrtTotalNetInput(neuron.signal, validOutputData[neuronN]);
                }

                // hidden layer pd error
                for (neuronN in net.layer[1].neurons) {
                    var neuron = net.layer[1].neurons[neuronN];
                    var erHiddenN = 0;

                    for (var h in net.layer[2].neurons) {
                        var outPutNeuron = net.layer[2].neurons[h];
                        erHiddenN += outPutNeuron.errorRate * outPutNeuron.input[neuronN].weight;
                    }

                    neuron.errorRate = erHiddenN * (neuron.signal * (1 - neuron.signal));
                }


                // output layer updateWeight
                for (neuronN in net.layer[2].neurons) {
                    var neuron = net.layer[2].neurons[neuronN];

                    for (var inX in neuron.input) {
                        var inP = neuron.input[inX];

                        pd_error_wrt_weight = neuron.errorRate * inP.weight;

                        inP.weight -= learningOptions.LEARNINGRATE * pd_error_wrt_weight
                    }


                }

                // hidden layer updateWeight
                for (neuronN in net.layer[1].neurons) {
                    var neuron = net.layer[1].neurons[neuronN];
                    for (var inX in neuron.input) {
                        var inP = neuron.input[inX];

                        pd_error_wrt_weight = neuron.errorRate * inP.weight;

                        inP.weight -= learningOptions.LEARNINGRATE * pd_error_wrt_weight
                    }
                }

                /*
                
                                for (i = net.layer.length - 1; i >= 1; i--) { // begin at the end
                                    currentL = net.layer[i];
                
                                    currentL.errorRate = getErrorPerLayer(currentL.neurons, outputValidperLayer);
                
                                    for (neuronN in currentL.neurons) {
                                        neuron = currentL.neurons[neuronN];
                
                                        var pd_error_wrt_weight;
                                        var erHiddenN = 0;
                
                                        if (outputlayer) {
                                            neuron.errorRate = getErrWrtTotalNetInput(neuron.signal, validOutputData[neuronN]);
                
                                            for (var inX in neuron.input) {
                                                var inP = neuron.input[inX];
                 
                                                pd_error_wrt_weight = neuron.errorRate * inP.weight;
                
                                                inP.weight -= learningOptions.LEARNINGRATE * pd_error_wrt_weight
                                            }
                
                                        }
                
                                        if (!outputlayer) { //must be a hidden layer then
                                            var laterLayer = net.layer[i + 1].neurons,
                                                laterNeuron = laterLayer[neuronN];
                                    
                                            for (var h in laterLayer) {
                                                if (typeof laterNeuron !== 'undefined')
                                                    erHiddenN += laterLayer[neuronN].errorRate * neuron.input[h].weight;
                                            }
                
                                            neuron.pderrors = erHiddenN * (neuron.signal * (1 - neuron.signal));
                                            
                                        
                                             for (var inX in neuron.input) {
                                                var inP = neuron.input[inX];
                 
                                                pd_error_wrt_weight = neuron.pderrors * inP.weight;
                
                                                inP.weight -= learningOptions.LEARNINGRATE * pd_error_wrt_weight
                                            }
                
                
                                            if (typeof laterNeuron !== 'undefined') {
                                                for (var h in laterLayer) {
                
                                                    var errorsWrtWeight = laterNeuron.errorRate * laterNeuron.input[h].signal;
                
                                                    laterNeuron.input[neuronN].weight -= learningOptions.LEARNINGRATE * errorsWrtWeight;
                
                                                }
                                            }
                
                                        }
                
                
                
                
                                    }
                                    //for deeper nets we would need to pass new estimated optimal output
                                    // here we might run into the vanishing gradient problem?
                                   // outputValidperLayer = [123];  // calculate new closer to valid value and set it for the next iteration
                                    outputlayer = false;
                                }*/








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
                this.signal = this.input[0].signal;
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

                this.signal = 1 / (1 + Math.exp(-sum)); // where Math.E is 2.718281828459045, why do we calculate with 2.7...?

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

            for (var itera = 0; itera < learningOptions.ITERATIONS; itera++) {
                var set = Math.floor(Math.random() * newInputData.length);
                learnFn(net, newInputData[set], validOutputData[set], learningOptions, itera);
            }
            net.done.call(net, { in: newInputData, out: validOutputData, err: net.layer.slice(-1)[0].errorRate });
        };

        return net;
    };


    return corpus;





};


//testing the magic//////////////////////////////////////////////////////////

var nnetFactory = new corpusCere(),
    learningOptions = { LEARNINGRATE: 0.001, ITERATIONS: 9403 };


var trainingset = [
    [1, 1],
    [1, 0],
    [1, 0.5]
    //  [1, 1]
];

var resultset = [
    [1],
    [0],
    [0]


    //  [0]
];


var net = nnetFactory.net("testNet")
    .addLayer("inputLayer", nnetFactory.growNeuronsRandom(2, "input", 0))
    .addLayer("hiddenLayer", nnetFactory.growNeuronsRandom(2, "sigmoid", 1))
    .addLayer("outputLayer", nnetFactory.growNeuronsRandom(1, "sigmoid", 1));

net.stimulus([1, 0]);

net.doneIteration = function (params) {
    this.stimulus(params.in);
    console.log("iteration: " + params.it, params.in, params.out, params.result, " errRate: " + params.err);

};

net.learn(trainingset, resultset, "backpropagation", learningOptions);

//no training [1, 0] = [0, 1] 0.8637949432782007 0.8154183737169318
//with training [1, 0] = [0, 1] 0.0020680089562570433 0.975465300792186

for (var s in trainingset) {
    var result = net.stimulus(trainingset[s]);

    console.log("Ergebnis: ", trainingset[s], JSON.stringify(result.neurons[0].signal));
}


process.exit();