//Implementation based off of:
//http://scikit-learn.org/stable/modules/naive_bayes.html
function BayesClassifier() {
	this.numWords = {
		true: 0,
		false: 0,
		'all': 0
	};
	this.db = {
		true: {},
		false: {},
		'all': {}
	};
	this.vocabulary = {};
	this.vocabularySize = 0;
}

BayesClassifier.prototype.normalizeInputToList = function(input) {
	input = input.toLowerCase();
	//remove non alphanumeric characters
	input = input.replace(/([^\w])+/g, ' ')
		input = input.trim().split(/\s+/);
	for (var i = input.length; i >= 0; i--) {
		if (input[i] === null || 
				input[i] === undefined || 
				input[i] === "") {
			input.splice(i, 1);
		}
	}
	return input;
}

BayesClassifier.prototype.train = function(input, type) {
	input = this.normalizeInputToList(input);

	for (var i = 0, l = input.length; i < l; i++) {
		if (!this.vocabulary[input[i]]) {
			this.vocabulary[input[i]] = true;
			this.vocabularySize++;
		}
		if (this.db[type][input[i]] === undefined) {
			this.db[type][input[i]] = 0;
		}
		this.db[type][input[i]]++;
		if (this.db.all[input[i]] === undefined) {
			this.db.all[input[i]] = 0;
		}
		this.db.all[input[i]]++;
		this.numWords[type]++;
		this.numWords.all++;
	}
}

//Lidstone smoothing
BayesClassifier.ALPHA = 0.75;
BayesClassifier.prototype.calculateProbability = function(word, type) {
	return ((this.db[type][word] || 0) + BayesClassifier.ALPHA) /
		((this.numWords[type] || 0) + BayesClassifier.ALPHA * this.vocabularySize);
}

BayesClassifier.prototype.calculateProbabilityAllWords = function(inputList, type) {
	var sum = 0;
	for (var i = 0; i < inputList.length; i++) {
		sum += Math.log(this.calculateProbability(inputList[i], type));
	}
	return sum;
}

BayesClassifier.prototype.guess = function(input, cutoff) {
	input = this.normalizeInputToList(input);
	var probtrue = this.calculateProbabilityAllWords(input, true);
	var probfalse = this.calculateProbabilityAllWords(input, false);
	//cutoff to adjust to favor one side.
	if (probtrue > (probfalse + cutoff)) {
		return true;
	}
	return false;
}
