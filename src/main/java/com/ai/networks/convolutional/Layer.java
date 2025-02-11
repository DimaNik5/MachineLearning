package com.ai.networks.convolutional;

public class Layer {
    protected Matrix[] inputs;
    protected Matrix[] temp;
    protected Matrix[] outputs;
    protected Filter[] filters;

    public void counting() {
        for(int i = 0; i < filters.length; i++){
            filters[i].convolution(inputs, temp[i]);
            temp[i].normalize();
            temp[i].pulling(outputs[i]);
        }
    }

    public Matrix[] getOut() {
        return outputs;
    }
}
