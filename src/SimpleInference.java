import java.util.*;

/**
 * Implements the Simple Inference algorithm (by full enumeration) for Bayesian Networks.
 * Computes P(queryVar = queryValue | evidence) by summing over all possible hidden variable assignments.
 */
public class SimpleInference implements InferenceAlgorithm {

    // --- Fields ---
    private BayesianNetwork bn;
    private int multiplicationCount = 0;
    private int additionCount = 0;

    // --- Constructor ---

    /**
     * Constructs a SimpleInference object for a given Bayesian network.
     * @param bn the Bayesian network to use.
     */
    public SimpleInference(BayesianNetwork bn) {
        this.bn = bn;
    }

    // --- Getters ---

    /**
     * @return the number of multiplications performed.
     */
    public int getMultiplicationCount() {
        return multiplicationCount;
    }

    /**
     * @return the number of additions performed.
     */
    public int getAdditionCount() {
        return additionCount;
    }

    // --- Operation Tracking ---

    @Override
    public void incrementMultiplications() {
        multiplicationCount++;
    }

    @Override
    public void incrementAdditions() {
        additionCount++;
    }

    // --- Inference Method ---

    /**
     * Computes the conditional probability P(queryVar = queryValue | evidence).
     * If all parents of the queried variable are present in the evidence, retrieves the value directly from the CPT.
     * Otherwise performs full enumeration over hidden variables.
     *
     * @param queryVar the name of the variable to query.
     * @param queryValue the desired value of the query variable.
     * @param evidence a map of observed variables and their values.
     * @return the conditional probability.
     */
    public double query(String queryVar, String queryValue, Map<String, String> evidence) {
        multiplicationCount = 0;
        additionCount = 0;

        Variable queryVariable = bn.getVariable(queryVar);
        List<String> parents = queryVariable.getParents();

        // Direct CPT lookup optimization
        if (evidence.keySet().containsAll(parents) && parents.containsAll(evidence.keySet())) {
            Map<String, String> full = new HashMap<>(evidence);
            full.put(queryVar, queryValue);
            return queryVariable.getProbability(full, bn.getVariableMap());
        }

        // Identify hidden variables
        List<String> hiddenVars = new ArrayList<>();
        for (Variable var : bn.getVariablesInTopologicalOrder()) {
            String name = var.getName();
            if (!evidence.containsKey(name) && !name.equals(queryVar)) {
                hiddenVars.add(name);
            }
        }

        List<Map<String, String>> hiddenAssignments = generateAllAssignments(hiddenVars, new HashMap<>());

        double numerator = 0.0;
        double denominator;

        // Compute numerator: P(queryVar=queryValue, evidence)
        Map<String, String> baseAssignment = new HashMap<>(evidence);
        baseAssignment.put(queryVar, queryValue);

        for (int i = 0; i < hiddenAssignments.size(); i++) {
            Map<String, String> extendedAssignment = new HashMap<>(baseAssignment);
            extendedAssignment.putAll(hiddenAssignments.get(i));
            double prob = computeJointProbability(extendedAssignment);
            numerator += prob;
            if (i > 0) incrementAdditions();
        }

        // Start denominator with numerator
        denominator = numerator;

        // Compute denominator: sum over all values of queryVar
        for (String value : queryVariable.getOutcomes()) {
            if (value.equals(queryValue)) continue;

            double sum = 0.0;
            Map<String, String> altAssignment = new HashMap<>(evidence);
            altAssignment.put(queryVar, value);

            for (int i = 0; i < hiddenAssignments.size(); i++) {
                Map<String, String> currentAlt = new HashMap<>(altAssignment);
                currentAlt.putAll(hiddenAssignments.get(i));
                double prob = computeJointProbability(currentAlt);
                sum += prob;
                if (i > 0) incrementAdditions();
            }

            incrementAdditions(); // sum added to denominator
            denominator += sum;
        }

        return numerator / denominator;
    }

    // --- Internal Helpers ---

    /**
     * Computes the joint probability for a full assignment of variables.
     * @param assignment a complete variable-value assignment.
     * @return the joint probability of that assignment.
     */
    private double computeJointProbability(Map<String, String> assignment) {
        double probability = 1.0;
        boolean first = true;

        for (Variable var : bn.getVariablesInTopologicalOrder()) {
            String varName = var.getName();
            if (!assignment.containsKey(varName)) continue;

            double prob = var.getProbability(assignment, bn.getVariableMap());
            probability *= prob;
            if (!first) incrementMultiplications();
            first = false;
        }

        return probability;
    }

    /**
     * Recursively generates all possible assignments for the given list of hidden variables.
     * @param hiddenVars list of variables to assign.
     * @param baseAssignment base assignment to extend.
     * @return a list of all possible full assignments to hiddenVars.
     */
    private List<Map<String, String>> generateAllAssignments(List<String> hiddenVars, Map<String, String> baseAssignment) {
        List<Map<String, String>> result = new ArrayList<>();
        if (hiddenVars.isEmpty()) {
            result.add(new HashMap<>(baseAssignment));
            return result;
        }

        String firstVar = hiddenVars.get(0);
        List<String> restVars = hiddenVars.subList(1, hiddenVars.size());

        for (String value : bn.getVariable(firstVar).getOutcomes()) {
            Map<String, String> extended = new HashMap<>(baseAssignment);
            extended.put(firstVar, value);
            result.addAll(generateAllAssignments(restVars, extended));
        }

        return result;
    }
}
