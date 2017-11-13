# building my own artificial neural network 

using sigmoid neurons and multiple layers

https://github.com/kindziora/corpusCere/blob/master/web/js/neuronal.js

just to get a deeper understanding of ai


```javascript

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
```
