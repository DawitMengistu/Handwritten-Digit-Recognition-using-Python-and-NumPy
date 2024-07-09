import fs from 'fs'
import * as math from 'mathjs';

class LayerDense {
    constructor(weights, biases) {
        this.weights = weights;
        this.biases = biases;
    }

    forward(inputs) {
        this.output = math.add(math.multiply(inputs, this.weights), this.biases);
    }
}

function sigmoid(x) {
    return math.dotDivide(1, math.add(1, math.exp(math.unaryMinus(x))));
}

// Apply the sigmoid function element-wise to a matrix
function sigmoidMatrix(matrix) {
    return math.map(matrix, x => sigmoid(x));
}



// Load the data from the JSON file
const data1 = fs.readFileSync('data/layer1_data.json', 'utf8');
const data2 = fs.readFileSync('data/layer2_data.json', 'utf8');

const LayerOneData = JSON.parse(data1);
const LayerTwo = JSON.parse(data2);

// Initialize the layer with the loaded data
const layer1 = new LayerDense(LayerOneData.weights, LayerOneData.biases);
const layer2 = new LayerDense(LayerTwo.weights, LayerTwo.biases);


export function getResult(X) {
    // Perform forward pass through the first layer
    layer1.forward(X);
    const activation1 = sigmoidMatrix(layer1.output);

    // Perform forward pass through the second layer
    layer2.forward(activation1);
    const outputLayer = sigmoidMatrix(layer2.output);

    return outputLayer[0];
}


