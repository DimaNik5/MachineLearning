package com.ai.networks.convolutional;

import com.ai.networks.Tokens;

import java.util.Arrays;

/**
 * Класс Filter хранит матрицы фильтра, а также значение смещения.
 * @author Никифоров Дмитрий
 * @since 1.1
 */
public class Filter {

    protected Matrix[] matrices;
    protected double biasValue;

    /**
     * Метод загрузки фильтра из строки
     * @param content {@code String}, содержащая информацию про фильтр.
     */
    public void loadFromString(String content){
        try{
            double[] cont = Arrays.stream(content.split(Tokens.SEP_OF_ELEMENTS)).mapToDouble(Double::parseDouble).toArray();
            matrices = new Matrix[(int)cont[0]];
            int count = (int)cont[1] * (int)cont[2];
            for(int i = 0; i < (int)cont[0]; i++){
                matrices[i] = new Matrix((int)cont[1], (int)cont[2]);
                for(int j = 0; j < count; j++){
                    matrices[i].addTo((int) (j / cont[2]), j % (int)cont[2], (int)cont[3 + count * i + j]);
                }
            }
            biasValue = cont[cont.length - 1];
        }catch (NumberFormatException e){
            throw new RuntimeException(e);
        }
    }

    /**
     * Свертка через конктретный фильтр.
     * <br>Примечание: количесво входных матриц должно совпадать с количеством матриц в фильтре.
     * @param input массив входных матриц.
     * @param output выходная матрица.
     */
    public void convolution(Matrix[] input, Matrix output){
        if(input.length != matrices.length) throw new RuntimeException("The number of input channels does not match the number of filter channels");
        output.setZero();
        for (int i = 0; i <= input[0].getH() - matrices[0].getH(); i++){
            for(int j = 0; j <= input[0].getW() - matrices[0].getW(); j++){
                for(int k = 0; k < input.length; k++){
                    output.addTo(i, j, overlay(i, j, input[k], matrices[k]));
                }
                output.addTo(i, j, biasValue);
            }
        }
    }

    /**
     * Метод для получения значения свертки по координатам.
     * @param i индекс по высоте.
     * @param j индекс по ширине.
     * @param in входная матрица
     * @param filter матрица фильтра
     * @return {@code double} - сумма произведений соответсвующих значений у входной матрицы и фильтра.
     */
    private double overlay(int i, int j, Matrix in, Matrix filter){
        double res = 0;
        for (int y = 0; y < filter.getH(); y++){
            for(int x = 0; x < filter.getW(); x++){
                res += filter.getCell(y, x) * in.getCell(i + y, j + x);
            }
        }
        return res;
    }
}
