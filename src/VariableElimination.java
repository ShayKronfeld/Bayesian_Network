import java.util.*;
import java.util.stream.Collectors;

/**
 * Implements the Variable Elimination algorithm for Bayesian network inference.
 * Computes P(queryVar = queryValue | evidence) using factor elimination.
 */
public class VariableElimination implements InferenceAlgorithm {

    // --- Fields ---
    private BayesianNetwork bn;
    private int multiplicationCount = 0;
    private int additionCount = 0;

    // --- Constructor ---
    /**
     * Constructs a new VariableElimination instance for a given Bayesian network.
     * @param bn the Bayesian network to perform inference on.
     */
    public VariableElimination(BayesianNetwork bn) {
        this.bn = bn;
    }

    // --- Getters ---
    /**
     * @return the number of multiplication operations performed during inference.
     */
    public int getMultiplicationCount() {
        return multiplicationCount;
    }

    /**
     * @return the number of addition operations performed during inference.
     */
    public int getAdditionCount() {
        return additionCount;
    }

    // --- Operation Counters ---
    @Override
    public void incrementMultiplications() {
        multiplicationCount++;
    }

    @Override
    public void incrementAdditions() {
        additionCount++;
    }

    // --- Main Inference API ---
    /**
     * Computes the conditional probability P(queryVar = queryValue | evidence).
     * @param queryVar the variable to query.
     * @param queryValue the value of the queried variable.
     * @param evidence observed evidence in the form of variable-value assignments.
     * @return the conditional probability.
     */
    public double query(String queryVar, String queryValue, Map<String, String> evidence) {
        multiplicationCount = 0;
        additionCount = 0;

        Set<String> relevant = findRelevantVariables(queryVar, evidence);

        if (canAnswerDirectlyFromCPT(queryVar, evidence)) {
            Map<String, String> fullAssignment = new HashMap<>(evidence);
            fullAssignment.put(queryVar, queryValue);
            return bn.getVariable(queryVar).getProbability(fullAssignment, bn.getVariableMap());
        }

        List<Factor> factors = new ArrayList<>();
        for (Variable var : bn.getVariables()) {
            if (relevant.contains(var.getName())) {
                Factor f = Factor.fromVariable(var, evidence, bn.getVariableMap());
                if (!f.getVariables().isEmpty() && !f.isTrivial(evidence)) {
                    factors.add(f);
                }
            }
        }

        List<String> hidden = relevant.stream()
                .filter(v -> !v.equals(queryVar) && !evidence.containsKey(v))
                .sorted()
                .collect(Collectors.toList());

        for (String toEliminate : hidden) {
            List<Factor> related = factors.stream()
                    .filter(f -> f.getVariables().contains(toEliminate))
                    .collect(Collectors.toList());

            related.sort(Comparator.comparingInt(this::asciiSum));

            factors.removeAll(related);

            if (related.isEmpty()) continue;

            Factor combined = joinFactorsInOrder(related, this);
            Factor reduced = combined.sumAndRemove(toEliminate, this);

            if (!reduced.isTrivial(evidence)) {
                factors.add(reduced);
            }
        }

        Factor finalFactor = joinFactorsInOrder(factors, this);

        double numerator = 0.0, denominator = 0.0;
        boolean seenNumerator = false, seenDenominator = false;

        for (Map.Entry<List<String>, Double> entry : finalFactor.getTable().entrySet()) {
            List<String> assignList = entry.getKey();
            Map<String, String> assign = new LinkedHashMap<>();
            for (int i = 0; i < finalFactor.getVariables().size(); i++) {
                assign.put(finalFactor.getVariables().get(i), assignList.get(i));
            }

            double val = entry.getValue();

            if (assign.get(queryVar).equals(queryValue)) {
                numerator += val;
                if (seenNumerator) incrementAdditions();
                seenNumerator = true;
            }

            denominator += val;
            if (seenDenominator) incrementAdditions();
            seenDenominator = true;
        }

        return numerator / denominator;
    }

    // --- Helpers ---

    /**
     * Joins all factors in the given list in order.
     * @param ordered list of factors already sorted by desired order.
     * @param counter counter for operation tracking.
     * @return the joined factor.
     */
    private Factor joinFactorsInOrder(List<Factor> ordered, InferenceAlgorithm counter) {
        if (ordered.isEmpty()) return null;
        Factor result = ordered.get(0);
        for (int i = 1; i < ordered.size(); i++) {
            result = result.join(ordered.get(i), counter);
        }
        return result;
    }

    /**
     * Computes the sum of ASCII values of all variable names in the factor.
     * Used for sorting factors deterministically.
     * @param f the factor to compute the sum for.
     * @return sum of ASCII values of all characters in variable names.
     */
    private int asciiSum(Factor f) {
        return f.getVariables().stream()
                .flatMapToInt(String::chars)
                .sum();
    }

    /**
     * Checks if a query can be answered directly using the CPT of the variable.
     * @param queryVar variable to query.
     * @param evidence evidence assignment.
     * @return true if direct CPT access is valid.
     */
    private boolean canAnswerDirectlyFromCPT(String queryVar, Map<String, String> evidence) {
        Variable var = bn.getVariable(queryVar);
        if (var.getParents().size() != evidence.size() || !var.getParents().containsAll(evidence.keySet())) {
            return false;
        }
        for (String ev : evidence.keySet()) {
            if (isDescendant(queryVar, ev)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Determines whether one variable is a descendant of another in the graph.
     * @param ancestor the potential ancestor variable.
     * @param descendant the variable to test.
     * @return true if descendant is in the subtree of ancestor.
     */
    public boolean isDescendant(String ancestor, String descendant) {
        return dfsIsDescendant(ancestor, descendant, new HashSet<>());
    }

    /**
     * Depth-first search for descendant relationship.
     */
    private boolean dfsIsDescendant(String current, String target, Set<String> visited) {
        if (visited.contains(current)) return false;
        visited.add(current);
        for (String child : bn.getChildren(current)) {
            if (child.equals(target) || dfsIsDescendant(child, target, visited)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Finds the set of relevant variables required for inference.
     * @param queryVar the target variable.
     * @param evidence evidence provided.
     * @return set of all relevant variable names.
     */
    private Set<String> findRelevantVariables(String queryVar, Map<String, String> evidence) {
        Set<String> relevant = new HashSet<>();
        Queue<String> queue = new LinkedList<>();

        relevant.add(queryVar);
        queue.add(queryVar);
        relevant.addAll(evidence.keySet());
        queue.addAll(evidence.keySet());

        while (!queue.isEmpty()) {
            String current = queue.poll();
            Variable var = bn.getVariable(current);
            if (var != null) {
                for (String parent : var.getParents()) {
                    if (relevant.add(parent)) {
                        queue.add(parent);
                    }
                }
            }
        }
        return relevant;
    }
}
