package com.ai.networks.convolutional;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;

public class Matrix {
    protected double[][] matrix;



    public static final byte RED_MASK = (byte) 0xFF0000;
    public static final byte GREEN_MASK = (byte) 0xFF00;
    public static final byte BLUE_MASK = (byte) 0xFF;

    public static final byte[] RGB_CHANNEL = {RED_MASK, GREEN_MASK, BLUE_MASK};
    public static final byte[] WB_CHANNEL = {RED_MASK | GREEN_MASK | BLUE_MASK};


    public Matrix(){}

    public Matrix(int height, int width){
        matrix = new double[height][width];
    }

    public void loadFromFile(String fileImage, byte mask){
        try{
            BufferedImage image = ImageIO.read(new File(fileImage));
            if(matrix == null) matrix = new double[image.getHeight()][image.getWidth()];
            else if(image.getHeight() != matrix.length || image.getWidth() != matrix[0].length){
                throw new RuntimeException("The image does not match the size");
            }
            for(int i = 0; i < image.getHeight(); i++){
                for(int j = 0; j < image.getWidth(); j++){
                    matrix[i][j] = image.getRGB(j, i) & mask;
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static Matrix[] getFromImage(String fileImage, byte[] channels){
        Matrix[] res = new Matrix[channels.length];
        for (int i = 0; i < res.length; i++){
            res[i] = new Matrix();
            res[i].loadFromFile(fileImage, channels[i]);
        }
        return res;
    }

    public void copyMatrix(double[][] matrix){
        if(this.matrix == null){
            this.matrix = new double[matrix.length][matrix[0].length];
        }
        else if(this.matrix.length != matrix.length || this.matrix[0].length != matrix[0].length){
            throw new RuntimeException("The matrix does not match the size");
        }
        for (int i = 0; i < matrix.length; i++){
            System.arraycopy(matrix[i], 0, this.matrix[i], 0, matrix[0].length);
        }
    }

    /**
     * sigmoid - метод активации нейрона, представленный в виде функции сигмоиды
     * @param x значения для активации
     * @return нормализованное значение
     */
    protected double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public void pulling(Matrix result){
        double[] values = new double[4];
        for(int i = 0; i < this.getH(); i += 2){
            for(int j = 0; j < this.getW(); j += 2){
                values[0] = matrix[i][j];
                if(i < this.getH() - 1){
                    values[1] = matrix[i + 1][j];
                    if(j < this.getW() - 1) values[3] = matrix[i + 1][j + 1];
                    else values[3] = -1;
                }
                else values[1] = -1;
                if(j < this.getW() - 1) values[2] = matrix[i][j + 1];
                else values[2] = -1;
                result.matrix[i /2][j /2] = Arrays.stream(values).max().getAsDouble();
            }
        }
    }

    public int getH(){return matrix.length;}

    public int getW(){return  matrix[0].length;}

    public void addTo(int i, int j, double value){
        matrix[i][j] += value;
    }

    public void setZero(){
        Arrays.stream(Arrays.stream(matrix).toArray()).forEach(e -> e = 0);
    }

    public double getCell(int i, int j){
        return matrix[i][j];
    }

    public void normalize(){
        Arrays.stream(Arrays.stream(matrix).toArray()).forEach(e -> e = sigmoid((Double) e));
    }
}
