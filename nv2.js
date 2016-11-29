var network = function () {
    var self = this;


    


    self.growNeuron = function (threshold, activationMethod) {
        return {
            input: [{}], //input weights and signals for training reasons
            threshold: threshold,
            activate: activationMethod
        };
    };

    self.addLayer = function (name, neurons) {
        //create weight connections to previous layer
        var connectionsCnt = net.layer.length > 0 ? net.layer.slice(-1)[0].neurons.length : 0;

        for (var current in neurons) {
            corpus.growConnections(neurons[current], connectionsCnt);
        }

        var level = { name: name, neurons: neurons };
        net.layer.push(level);
        return net;
    };


};