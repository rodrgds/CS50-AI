import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    conditional_prob = 1
    people_to_calculate = list(people.keys())
    calculated_people = dict()
    mu = PROBS["mutation"]
    while len(people_to_calculate) != 0:
        person = people_to_calculate[0]
        eval_for_gene = 1 if person in one_gene else 2 if person in two_genes else 0
        eval_for_trait = person in have_trait

        if not people[person]["mother"] and not people[person]["father"]:
            # no parents listed
            calculated_people[person] = PROBS["gene"][eval_for_gene]
        else:
            # we need to first maybe sort the people dict based on the available data/dependencies (first calculate the ones with no parents, etc. etc.)
            mother, father = (people[person][p] for p in ["mother", "father"])
            parents = [mother, father]
            if not all(parent in calculated_people for parent in parents):
                people_to_calculate.remove(person)
                people_to_calculate.append(person)
                continue

            mother_g = 1 if mother in one_gene else 2 if mother in two_genes else 0
            father_g = 1 if father in one_gene else 2 if father in two_genes else 0
            parent_gs = (mother_g, father_g)
            # TODO: talvez tirar 1-mu do 1 (0.5)
            probs = [mu, 0.5, 1-mu]
            prob1, prob2 = (probs[g] for g in parent_gs)
            results = [(1-prob1) * (1-prob2), prob1*(1-prob2) + prob2*(1-prob1), prob1 * prob2]
            calculated_people[person] = results[eval_for_gene]

        calculated_people[person] *= PROBS["trait"][eval_for_gene][eval_for_trait]
        # print("COND PROB OF", person, calculated_people[person])

        conditional_prob *= calculated_people[person]
        people_to_calculate.remove(person)

    return conditional_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        eval_for_gene = 1 if person in one_gene else 2 if person in two_genes else 0
        eval_for_trait = person in have_trait
        probabilities[person]["gene"][eval_for_gene] += p
        probabilities[person]["trait"][eval_for_trait] += p

    return


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # gene
        genes = probabilities[person]["gene"]
        total = sum(genes[i] for i in genes)
        for i in genes:
            probabilities[person]["gene"][i] /= total
        # traits
        traits = probabilities[person]["trait"]
        total = traits[True] + traits[False]
        for val in [True, False]:
            probabilities[person]["trait"][val] /= total



if __name__ == "__main__":
    main()
