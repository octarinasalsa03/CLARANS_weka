package weka.classifiers.functions;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.util.*;

public class GLVQ extends AbstractClassifier implements OptionHandler {

    private class Pair<T>{
        protected T k;
        protected T v;

        public Pair(T k, T v){
            this.k = k;
            this.v = v;
        }
    }

    private double codebooks[][][];
    private int numClass;
    private int numFeatures;
    private int numCodebook;
    private double learningRate;
    private int epoch;

    public GLVQ(){
        numCodebook = 1;
        learningRate = 0.001;
        epoch = 100;
    }

    public String globalInfo() {
        return "Class for using GLVQ algorithm for prediction. Support Multicodebook "
                + "for class representation";
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.STRING_CLASS);
        return result;
    }

    /**
     *
     * Parses a given list of options.
     * <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -N
     *  Set the number of codebooks per class, integer.
     *  (default = 1)</pre>
     *
     * <pre> -L
     *  The learning rate. If normalization is
     *  turned off (as it is automatically for streaming data), then the
     *  default learning rate will need to be reduced (try 0.0001).
     *  (default = 0.001).</pre>
     *
     * <pre> -E &lt;integer&gt;
     *  The number of epochs to perform (batch learning only, default = 500)</pre>

     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        String nCodebook = Utils.getOption('N', options);
        if (nCodebook.length() > 0) {
            setNumCodebook(Integer.parseInt(nCodebook));
        }

        String lRate = Utils.getOption('L', options);
        if (lRate.length() > 0) {
            setLearningRate(Double.parseDouble(lRate));
        }

        String epochsString = Utils.getOption("E", options);
        if (epochsString.length() > 0) {
            setEpoch(Integer.parseInt(epochsString));
        }

        super.setOptions(options);
    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<String>();

        options.add("-N");
        options.add("" + getNumCodebook());
        options.add("-L");
        options.add("" + getLearningRate());
        options.add("-E");
        options.add("" + getEpoch());

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[1]);
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>();
        newVector.add(new Option("\tSet the Codebook Number per class.\n\t"
                , "N", 1, "-N"));
        newVector
                .add(new Option(
                        "\tThe learning rate. (default = 0.001).", "L", 1, "-L"));
        newVector.add(new Option("\tThe number of epochs to perform ("
                + "batch learning only, default = 100)", "E", 1, "-E <integer>"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    /**
     * Generates a linear regression function predictor.
     *
     * @param argv the options
     */
    public static void main(String argv[]) {
        runClassifier(new GLVQ(), argv);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        int numInstance = data.numInstances();
        this.numFeatures = data.numAttributes() - 1; //minus 1 since one among them is class attribute
        this.numClass = data.numClasses();

        // init Codebook vector
        initCodebooks();

        //loop epoch
        for (int i = 0; i < getEpoch(); i++) {
            //loop data
            for (int j = 0; j < numInstance; j++) {
                Instance instance = data.get(j);
                double features[] = extractFeatures(instance);
                int target = (int) instance.classValue();
                System.out.println("class: " + target);
                updateCodebooks(features, target, i);
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        double distances[][] = new double[this.numClass][getNumCodebook()];
        double X[] = extractFeatures(instance);
        double probs[] = new double[this.numClass];
        for (int i = 0; i < this.numClass; i++) {
            for (int j = 0; j < getNumCodebook(); j++) {
                distances[i][j] = euclidDistance(X, codebooks[i][j]);
            }
        }

        Pair<Integer> aMin = argmin2(distances);
        probs[aMin.k] = 1;
        return probs;
    }

    private void initCodebooks(){
        codebooks = new double[this.numClass][getNumCodebook()][this.numFeatures];
        // loop each codebook to init Random Values
        for (int i = 0; i < this.numClass; i++) {
            for (int j = 0; j < getNumCodebook(); j++) {
                for (int k = 0; k < this.numFeatures; k++) {
                    codebooks[i][j][k] = new Random().nextDouble()*100;
                }
            }
        }
    }

    private double[] extractFeatures(Instance instance){
        //mark the class index
        double features[] = new double[this.numFeatures];
        int j = 0;
        for (int i = 0; i < instance.numAttributes(); i++) {
            if(i != instance.classIndex()){
                features[j] = instance.value(i);
                j++;
            }
        }
        return features;
    }

    private void updateCodebooks(double X[], int target, int epoch){
        Pair<Integer> pos1 = findW1Pos(X, target);
        Pair<Integer> pos2 = findW2Pos(X, target);
        double w1[] = codebooks[pos1.k][pos1.v];
        double w2[] = codebooks[pos2.k][pos2.v];
        double d1 = euclidDistance(X, w1);
        double d2 = euclidDistance(X, w2);
        double fmu_deriv = fmuDeriv(d1, d2, epoch);
        double mu_d1_deriv = d2 / Math.pow(d1 + d2,2);
        double mu_d2_deriv = d1 / Math.pow(d1 + d2,2);

        double d1_w1_deriv[] = arrayMinus(X, w1);
        double d2_w2_deriv[] = arrayMinus(X, w2);

        double w1_delta[] = arrayMultiplyByConstant(d1_w1_deriv,fmu_deriv * mu_d1_deriv);
        double w2_delta[] = arrayMultiplyByConstant(d2_w2_deriv, fmu_deriv * mu_d2_deriv);

        w1 = arrayPlus(w1,arrayMultiplyByConstant(w1_delta, this.learningRate));
        w2 = arrayMinus(w2, arrayMultiplyByConstant(w2_delta, this.learningRate));

        //rewrite to codebooks
        codebooks[pos1.k][pos1.v] = w1;
        codebooks[pos2.k][pos2.v] = w2;
    }

    private double[] arrayMultiplyByConstant(double X[], double K){
        double R[] = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            R[i] = X[i] * K;
        }
        return R;
    }

    private double[] arrayPlus(double X[], double Y[]){
        double Z[] = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            Z[i] = X[i] + Y[i];
        }
        return Z;
    }

    private double[] arrayMinus(double X[], double Y[]){
        double Z[] = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            Z[i] = X[i] - Y[i];
        }
        return Z;
    }

    private double fmu(double d1, double d2, int epoch){
        double muval = computeMu(d1, d2);
        return 1 / (1 + Math.exp(-muval * epoch));
    }

    private double fmuDeriv(double d1, double d2, int epoch){
        double fmuval = fmu(d1, d2, epoch);
        return fmuval * (1-fmuval);
    }

    private double computeMu(double d1, double d2){
        return (d1-d2)/(d1+d2);
    }

    private Pair<Integer> findW1Pos(double X[], int target){
        double distance[] = new double[getNumCodebook()];
        for (int i = 0; i < getNumCodebook(); i++) {
            distance[i] = euclidDistance(X, codebooks[target][i]);
        }
        int aMin = argmin(distance);
        return new Pair(target,aMin);
    }

    private Pair<Integer> findW2Pos(double X[], int target){
        double[][] distance = new double[this.numClass][getNumCodebook()];
        // set current class distance to High values
        for (int i = 0; i < getNumCodebook(); i++) {
            distance[target][i] = 2E32;
        }

        for (int i = 0; i < this.numClass; i++) {
            for (int j = 0; j < getNumCodebook(); j++) {
                if(i != target)
                    distance[i][j] = euclidDistance(X, codebooks[i][j]);
            }
        }
        Pair<Integer> aMin = argmin2(distance);
        return aMin;
    }

    private Pair<Integer> argmin2(double val[][]){
        double mval = val[0][0];
        int icol = 0;
        int irow =0;
        for (int i = 0; i < this.numClass; i++) {
            for (int j = 0; j < getNumCodebook(); j++) {
                if(val[i][j] < mval){
                    mval = val[i][j];
                    irow = i;
                    icol = j;
                }
            }
        }
        return new Pair(irow, icol);
    }

    private int argmin(double val[]){
        if(val.length==1)
            return 0;
        double mval = val[0];
        int ival = 0;
        for (int i = 0; i < val.length; i++) {
            if(val[i] < mval) {
                mval = val[i];
                ival = i;
            }
        }
        return ival;
    }

    private double euclidDistance(double[] X, double[] Y){
        /**
         * Assume X and & of the same attributes
         * Find norm-2 distance between two vectors
         */
        double result = 0;
        for (int i = 0; i < X.length; i++) {
            result += (X[i] - Y[i])*(X[i] - Y[i]);
        }
        return Math.sqrt(result);
    }

    public int getNumCodebook() {
        return numCodebook;
    }

    public void setNumCodebook(int numCodebook) {
        this.numCodebook = numCodebook;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }
}