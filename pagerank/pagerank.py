import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # dict for each page, filled with 0s
    result_dict = { p:0 for p in corpus }
    page_number = len(result_dict)
    default_odds = 1 / page_number
    if corpus[page] == set():
        # equal distro for all
        for p in result_dict:
           result_dict[p] = default_odds 
    else:
        for p in result_dict:
            result_dict[p] = (1-damping_factor) * default_odds
            
        # now times by 1-d
        outgoing_odds = 1 / len(corpus[page])
        for p in corpus[page]:
            result_dict[p] += damping_factor * outgoing_odds

    return result_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page = random.choice(list(corpus.keys()))
    total_occurrences = { p:0 for p in corpus }
    for _ in range(n):
        next_odds = transition_model(corpus, page, damping_factor)
        
        # choose based on the odds, the next page at random
        choice = random.random()
        total = 0
        for p in next_odds:
            total += next_odds[p]
            if choice <= total:
                page = p
                total_occurrences[page] += 1
                break
    return {p:total_occurrences[p]/n for p in total_occurrences}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    CONVERGENCE_CONSTANT = 0.001
    N = len(corpus)
    page_ranks = { p:1/N for p in corpus }
    
    incoming_links = { p:set() for p in corpus }
    for page in corpus:
        if corpus[page] == set():
            corpus[page] = list(corpus.keys())
        for p in corpus[page]:
            incoming_links[p].add(page)
    
    while True:
        new_page_ranks = page_ranks.copy()
        
        for page in page_ranks:
            s = 0
            for p in incoming_links[page]:
                s += page_ranks[p] / len(corpus[p])
            new_page_ranks[page] = (1-damping_factor)/N + damping_factor*s
        
        if all([abs(page_ranks[page] - new_page_ranks[page]) < CONVERGENCE_CONSTANT for page in page_ranks]):
            break
                
        page_ranks = new_page_ranks
        
    return page_ranks

if __name__ == "__main__":
    main()
