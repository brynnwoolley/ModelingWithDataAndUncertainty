"""
Information Theory Lab

Brynn Woolley
Section 002
"""


import numpy as np
import wordle

# Problem 1
def get_guess_result(guess, true_word):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the secret word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        guess (string) - the guess being made
        true_word (string) - the secret word
    Returns:
        result (list of integers) - the result of the guess, as described above
    """
    # cast true_word to list
    trueWord = list(true_word)
    result = [0,0,0,0,0]
    # iterate through guess & construct list
    for i in range(5):
        if guess[i] == true_word[i]:
            result[i] = 2
            trueWord.remove(guess[i])

    for i in range(5):
        if guess[i] in trueWord and guess[i] != true_word[i]:
            result[i] = 1
            trueWord.remove(guess[i])

    return result

# Helper function
def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they 
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words
    
# Problem 2
def compute_highest_entropy(all_guess_results='all_guess_results.npy', allowed_guesses='allowed_guesses.txt'):
    """
    Compute the entropy of each allowed guess.
    
    Arguments:
        all_guess_results ((n,m) ndarray) - the array found in
            all_guess_results.npy, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_guesses (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
    """
    # Load data
    if isinstance(all_guess_results, str):              # load in npy if one is given
        all_guess_results = np.load(all_guess_results)
    else:
        all_guess_results = np.array(all_guess_results) # else convert whatever into an ndarry 
    
    if isinstance(allowed_guesses, str):
        allowed_guesses = load_words(allowed_guesses)   # similar to above
    else:
        allowed_guesses = list(allowed_guesses)
    n = len(allowed_guesses)
    entropies = np.zeros(n)

    # calc entropy for each guess
    for i in range(n):
        results = all_guess_results[i]                          # result for current guess

        counts = np.bincount(results)                           # count how many 0's, 1's, & 2's

        probs = counts/len(results)                             # calculate probability

        entropy =  -np.sum(probs * np.log2(probs + 1e-100))    # calc entropy & avoid dividing by zero
        entropies[i] = entropy
    
    best_guess = allowed_guesses[np.argmax(entropies)]          # find guess w/ highest entropy
    
    return best_guess

# Problem 3
def filter_words(guess, result, all_guess_results='all_guess_results.npy', allowed_guesses='allowed_guesses.txt', possible_secret_words='InformationTheory/possible_secret_words.txt'):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already have an array of the result of all guesses for all possible words, 
    we will use this array instead of recomputing the results.
    
	Return a filtered list of possible words that are still possible after 
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array will be used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (2-D ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        allowed_guesses (list of str)
            The list of words we are allowed to guess
        possible_secret_words (list of str)
            The list of possible secret words
        guess (str)
            The guess we made
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (2-D ndarray) The filtered array of guess results
    """
    filtered_words = []

    # use base 3 to cast the result to an integer
    if isinstance(result, (list,tuple)) and len(result) == 5:
        result = sum((3**i)*result[i] for i in range(5))

    guess_i = allowed_guesses.index(guess)

    # loop through possible words
    for i in range(len(possible_secret_words)):
        curr_result = all_guess_results[guess_i, i]             # calc result for current guess

        if curr_result == result:                               # if it matches the result, include in filtered list
            filtered_words.append(possible_secret_words[i])

    # find indices & filter the array
    word_i = [i for i, word in enumerate(possible_secret_words) if word in filtered_words]
    filtered_results = all_guess_results[:, word_i]

    return filtered_words, filtered_results

# Problem 4
def play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    
    # loop until the game is completed
    while not game.is_finished():

        if len(possible_secret_words) == 1:                 # guess word if it's narrowed down to one
            guess = possible_secret_words[0]

        else:                                               # otherwise guess randomly (ignoring entropy)
            guess = np.random.choice(allowed_guesses)

        result, num_guesses = game.make_guess(guess)

        # use filter function to filter
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)

    return num_guesses

'''def test4():
    data = np.load('all_guess_results.npy')
    allowed_guesses = load_words('allowed_guesses.txt')
    possible_secret_words = load_words('possible_secret_words.txt')
    game = wordle.WordleGame()
    print(play_game_naive(game, data, possible_secret_words, allowed_guesses, word='excel', display=True))'''

# Problem 5
def play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    
    # loop until the game is completed
    while not game.is_finished():
        if len(possible_secret_words) == 1:                 # guess word if it's narrowed down to one
            guess = possible_secret_words[0]
        
        else:                                               # otherwise choose guess w/ highest entropy
            guess = compute_highest_entropy(all_guess_results, allowed_guesses)

        result, num_guesses = game.make_guess(guess)

        # use filter function to filter
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)

    return num_guesses

"""def test4_and_5():
    data = np.load('all_guess_results.npy')
    allowed_guesses = load_words('allowed_guesses.txt')
    possible_secret_words = load_words('possible_secret_words.txt')
    game = wordle.WordleGame()
    word = np.random.choice(possible_secret_words)
    print(play_game_naive(game, data, possible_secret_words, allowed_guesses, word=word, display=True))
    print(play_game_entropy(game, data, possible_secret_words, allowed_guesses, word='excel', display=True))
"""
# Problem 6
def compare_algorithms(all_guess_results, possible_secret_words, allowed_guesses, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.
    
    
    Arguments:
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy, 
            containing the result of making any allowed 
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
        # initialize empty lists to store guesses
    naive_guesses, entropy_guesses = [], []

    for i in range(n):
        # generate random word & copy the possible_secret_words file
        word = np.random.choice(possible_secret_words)  # generate random word
        possible_secret_words_i = possible_secret_words.copy()

        # naive Wordle
        game = wordle.WordleGame()
        naive_guess_count = play_game_naive(game, all_guess_results, possible_secret_words_i, allowed_guesses, word=word, display=False)
        naive_guesses.append(naive_guess_count)

        # entropy Wordle
        game = wordle.WordleGame()
        entropy_guess_count = play_game_entropy(game, all_guess_results, possible_secret_words_i, allowed_guesses, word=word, display=False)
        entropy_guesses.append(entropy_guess_count)

    # calc averages
    avg_naive = np.mean(naive_guesses)
    avg_entropy = np.mean(entropy_guesses)

    return avg_naive, avg_entropy

"""def test6():
    data = np.load('all_guess_results.npy')
    allowed_guesses = load_words('allowed_guesses.txt')
    possible_secret_words = load_words('possible_secret_words.txt')
    game = wordle.WordleGame()
    word = np.random.choice(possible_secret_words)
    print(compare_algorithms(data, possible_secret_words, allowed_guesses, n=5))"""
    
