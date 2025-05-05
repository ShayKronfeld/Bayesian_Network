import java.util.*;
import java.util.stream.Collectors;

/**
 * Implements the Heuristic Variable Elimination algorithm for Bayesian Networks.
 * Uses heuristics such as min-fill, factor size, and ASCII ordering to guide variable elimination.
 */
public class HeuristicVariableElimination implements InferenceAlgorithm {

    // --- Fields ---
    private BayesianNetwork bn;
    private int multiplicationCount = 0;
    private int additionCount = 0;

    // --- Constructor ---

    /**
     * Constructs a HeuristicVariableElimination object for a given Bayesian network.
     * @param bn the Bayesian network to use.
     */
    public HeuristicVariableElimination(BayesianNetwork bn) {
        this.bn = bn;
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

    /**
     * @return the total number of multiplications performed during inference.
     */
    public int getMultiplicationCount() {
        return multiplicationCount;
    }

    /**
     * @return the total number of additions performed during inference.
     */
    public int getAdditionCount() {
        return additionCount;
    }

    // --- Main Query Function ---

    /**
     * Computes the conditional probability P(queryVar = queryValue | evidence)
     * using heuristic-based variable elimination.
     *
     * @param queryVar the query variable.
     * @param queryValue the desired value of the query variable.
     * @param evidence observed variable assignments.
     * @return the computed conditional probability.
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

        // Construct initial set of factors
        List<Factor> factors = new ArrayList<>();
        for (Variable var : bn.getVariables()) {
            if (!relevant.contains(var.getName())) continue;

            Factor raw = Factor.fromVariable(var, evidence, bn.getVariableMap());
            Factor reduced = raw;

            for (Map.Entry<String, String> ev : evidence.entrySet()) {
                if (reduced.getVariables().contains(ev.getKey())) {
                    reduced = reduced.restrict(ev.getKey(), ev.getValue());
                }
            }

            if (!reduced.isEmpty() &&
                    (!reduced.getVariables().stream().allMatch(evidence::containsKey) || reduced.getVariables().contains(queryVar))) {
                factors.add(reduced);
            }
        }

        // Determine hidden variables
        List<String> hidden = relevant.stream()
                .filter(v -> !v.equals(queryVar) && !evidence.containsKey(v))
                .collect(Collectors.toList());

        // Heuristic elimination
        while (!hidden.isEmpty()) {
            String toEliminate = selectBestVar(hidden, factors);
            hidden.remove(toEliminate);

            if (toEliminate.equals(queryVar)) continue;

            List<Factor> related = factors.stream()
                    .filter(f -> f.getVariables().contains(toEliminate))
                    .collect(Collectors.toList());

            factors.removeAll(related);
            if (related.isEmpty()) continue;

            Factor joined = multiplyFactors(related);
            Factor summed = joined.sumAndRemove(toEliminate, this);

            if (!summed.getVariables().isEmpty()) {
                factors.add(summed);
            }
        }

        Factor finalFactor = multiplyFactors(factors);
        if (!finalFactor.getVariables().contains(queryVar)) {
            throw new IllegalStateException("Final factor is missing query variable: " + queryVar);
        }

        // Compute final probability
        double numerator = 0.0, denominator = 0.0;
        boolean first = true;
        int queryIdx = finalFactor.getVariables().indexOf(queryVar);

        for (Map.Entry<List<String>, Double> entry : finalFactor.getTable().entrySet()) {
            List<String> assignment = entry.getKey();
            double val = entry.getValue();
            if (assignment.get(queryIdx).equals(queryValue)) numerator += val;
            denominator += val;
            if (!first) incrementAdditions();
            first = false;
        }

        return numerator / denominator;
    }

    // --- Heuristic Variable Selection ---

    /**
     * Selects the best variable to eliminate based on multiple heuristics.
     */
    private String selectBestVar(List<String> vars, List<Factor> factors) {
        String bestVar = null;
        int bestSize = Integer.MAX_VALUE;
        int bestAsciiSum = Integer.MAX_VALUE;
        int bestParticipation = Integer.MAX_VALUE;
        int bestMaxFactorSize = Integer.MAX_VALUE;

        for (String candidate : vars) {
            List<Factor> involved = factors.stream()
                    .filter(f -> f.getVariables().contains(candidate))
                    .collect(Collectors.toList());

            Set<String> neighbors = involved.stream()
                    .flatMap(f -> f.getVariables().stream())
                    .collect(Collectors.toSet());
            neighbors.remove(candidate);

            int estimatedSize = neighbors.stream()
                    .mapToInt(n -> bn.getVariable(n).getOutcomes().size())
                    .reduce(1, (a, b) -> a * b);

            int asciiSum = candidate.chars().sum();
            int participation = involved.size();
            int maxFactorSize = involved.stream()
                    .mapToInt(f -> f.getVariables().size())
                    .max().orElse(0);

            if (
                    estimatedSize < bestSize ||
                            (estimatedSize == bestSize && participation < bestParticipation) ||
                            (estimatedSize == bestSize && participation == bestParticipation && maxFactorSize < bestMaxFactorSize) ||
                            (estimatedSize == bestSize && participation == bestParticipation && maxFactorSize == bestMaxFactorSize && asciiSum < bestAsciiSum) ||
                            (estimatedSize == bestSize && participation == bestParticipation && maxFactorSize == bestMaxFactorSize &&
                                    asciiSum == bestAsciiSum && (bestVar == null || candidate.compareTo(bestVar) < 0))
            ) {
                bestVar = candidate;
                bestSize = estimatedSize;
                bestAsciiSum = asciiSum;
                bestParticipation = participation;
                bestMaxFactorSize = maxFactorSize;
            }
        }

        return bestVar;
    }

    // --- Factor Operations ---

    /**
     * Multiplies a list of factors into a single factor using a priority queue based on size and ASCII.
     */
    private Factor multiplyFactors(List<Factor> factors) {
        PriorityQueue<Factor> queue = new PriorityQueue<>(Comparator
                .comparingInt((Factor f) -> f.getVariables().size())
                .thenComparingInt(f -> f.getVariables().stream().mapToInt(String::hashCode).sum()));

        queue.addAll(factors);
        while (queue.size() > 1) {
            Factor f1 = queue.poll();
            Factor f2 = queue.poll();
            queue.add(f1.join(f2, this));
        }
        return queue.poll();
    }

    // --- Helper Methods ---

    /**
     * Returns all relevant variables that influence the query or are part of the evidence.
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

    /**
     * Determines if the query can be directly answered from the CPT without full inference.
     */
    private boolean canAnswerDirectlyFromCPT(String queryVar, Map<String, String> evidence) {
        Variable var = bn.getVariable(queryVar);
        for (String parent : var.getParents()) {
            if (!evidence.containsKey(parent)) return false;
        }
        for (String ev : evidence.keySet()) {
            if (!var.getParents().contains(ev)) return false;
            if (isDescendant(queryVar, ev)) return false;
        }
        return true;
    }

    /**
     * Checks if `target` is a descendant of `ancestor` in the network.
     */
    private boolean isDescendant(String ancestor, String target) {
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.add(ancestor);
        visited.add(ancestor);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            for (Variable var : bn.getVariables()) {
                if (var.getParents().contains(current)) {
                    String child = var.getName();
                    if (child.equals(target)) return true;
                    if (visited.add(child)) {
                        queue.add(child);
                    }
                }
            }
        }
        return false;
    }
}
