/**
 * @author Alexander Kindziora
 * 
 */

// building my own artificial neural network using sigmoid neurons and multiple layers
// just to get a deeper understanding of ai

var corpusBrain = function () {
    var corpus = this;

    /**
     * 
     */
    corpus.cellBuilder = function () {
        var me = this;
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
    corpus.net = function () {
        var net = this;
        var layer = [];

        net.addLayer = function (name, neurons) {
            var level = { name: name, neurons: neurons };
            layer.push(level);
            return level;
        };

        net.stimulus = function (neurons) {

        };

        net.learn = function (neurons) {

        };

        
        return net;
    } ();
   
    return corpus;
};


//running the magic

var neuronFactory = new corpusBrain();

neuralNet.addLayer("inputLayer", neuronFactory.growNeuronsRandom(3));
neuralNet.addLayer("hiddenLayer", neuronFactory.growNeuronsRandom(3));
neuralNet.addLayer("outputLayer", neuronFactory.growNeuronsRandom(1));


neuralNet



