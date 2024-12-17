package networks;

public interface Teacher<IN, OUT> {

    int setData(IN[][] inputs, OUT[][] outputs);

    int addData(IN[][] inputs, OUT[][] outputs);

    int addData(IN[] inputs, OUT[] outputs);

    IN[][] getIn();

    OUT[][] getOut();

    boolean checkData();
}
