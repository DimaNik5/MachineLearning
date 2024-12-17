package networks.perceptron.train;

import networks.Teacher;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

public class PerceptronTeacher implements Teacher<Double, Double> {

    private Double[][] inputs;
    private Double[][] outputs;

    public int loadData(String fileIn, String fileOut){
        int c = 0;
        try(BufferedReader bfi = new BufferedReader(new FileReader(fileIn)); BufferedReader bfo = new BufferedReader(new FileReader(fileOut))){
            while(bfi.readLine() != null && bfo.readLine() != null){
                c++;
            }
        }catch (Exception e){
            return 1;
        }
        inputs = new Double[c][];
        outputs = new Double[c][];
        try(BufferedReader bfi = new BufferedReader(new FileReader(fileIn)); BufferedReader bfo = new BufferedReader(new FileReader(fileOut))){
            String li, lo;
            int j = 0;
            while((li = bfi.readLine()) != null && (lo = bfo.readLine()) != null){
                String[] str = li.split(",");
                if(j != 0) {
                    if(inputs[j - 1].length != str.length) continue;
                }
                String[] str1 = lo.split(",");
                if(j != 0) {
                    if(outputs[j - 1].length != str1.length) continue;
                }
                inputs[j] = new Double[str.length];
                for(int i = 0; i < str.length; i++){
                    inputs[j][i] = Double.parseDouble(str[i]);
                }

                outputs[j] = new Double[str1.length];
                for(int i = 0; i < str1.length; i++){
                    outputs[j][i] = Double.parseDouble(str1[i]);
                }
                j++;
            }
        }catch (Exception e){
            return 1;
        }
        return 0;
    }

    @Override
    public int setData(Double[][] inputs, Double[][] outputs) {
        if(inputs.length == outputs.length && inputs.length > 0){
            this.inputs = inputs;
            this.outputs = outputs;
            return 0;
        }
        return 1;
    }

    @Override
    public int addData(Double[][] inputs, Double[][] outputs) {
        if(inputs.length == outputs.length && inputs.length > 0){
            if(this.inputs == null) {
                this.inputs = inputs;
                this.outputs = outputs;
                return 0;
            }
            int t = this.inputs.length;
            this.inputs = Arrays.copyOf(this.inputs, t + inputs.length);
            System.arraycopy(inputs, 0, this.inputs, t, inputs.length);

            this.outputs = Arrays.copyOf(this.outputs, t + outputs.length);
            System.arraycopy(outputs, 0, this.outputs, t, outputs.length);
            return 0;
        }
        return 1;
    }

    @Override
    public int addData(Double[] inputs, Double[] outputs) {
        if(inputs.length > 0 && outputs.length > 0){
            if(this.inputs == null) {
                this.inputs = new Double[1][];
                this.inputs[0] = inputs;
                this.outputs = new Double[1][];
                this.outputs[0] = outputs;
                return 0;
            }

            this.inputs = Arrays.copyOf(this.inputs, this.inputs.length + 1);
            this.inputs[this.inputs.length - 1] = inputs;

            this.outputs = Arrays.copyOf(this.outputs, this.outputs.length + 1);
            this.outputs[this.outputs.length - 1] = outputs;
            return 0;
        }
        return 1;
    }

    @Override
    public Double[][] getIn() {
        return inputs;
    }

    @Override
    public Double[][] getOut() {
        return outputs;
    }

    @Override
    public boolean checkData() {
        return inputs.length == outputs.length && inputs.length > 0;
    }
}
