const allQuestions = [
  { question: "Tell me about yourself.", category: "Behavioral" },
  { question: "Describe a time you worked in a team.", category: "Behavioral" },
  { question: "How do you handle conflict at work?", category: "Behavioral" },
  { question: "Describe a time you failed. What did you learn?", category: "Behavioral" },
  { question: "What motivates you?", category: "Behavioral" },
  { question: "What are your strengths and weaknesses?", category: "Behavioral" },
  { question: "How do you prioritize tasks?", category: "Situational" },
  { question: "Tell me about a time you showed leadership.", category: "Situational" },
  { question: "How do you manage stress or pressure?", category: "Situational" },
  { question: "Give an example of a goal you set and achieved.", category: "Situational" },
  { question: "What is object-oriented programming?", category: "Technical" },
  { question: "Explain polymorphism with an example.", category: "Technical" },
  { question: "What is recursion?", category: "Technical" },
  { question: "Difference between list and tuple in Python.", category: "Technical" },
  { question: "What are the four pillars of OOP?", category: "Technical" },
  { question: "What is the difference between a stack and a queue?", category: "Technical" },
  { question: "Explain how a hash table works.", category: "Technical" },
  { question: "What is a binary search tree?", category: "Technical" },
  { question: "Time complexity of searching in an array.", category: "Technical" },
  { question: "When would you use a linked list?", category: "Technical" },
  { question: "What is normalization in databases?", category: "Database" },
  { question: "Difference between INNER JOIN and OUTER JOIN.", category: "Database" },
  { question: "What is a primary key and foreign key?", category: "Database" },
  { question: "How do indexes work in SQL?", category: "Database" },
  { question: "What are ACID properties?", category: "Database" },
  { question: "What is REST API?", category: "Web" },
  { question: "Difference between GET and POST.", category: "Web" },
  { question: "What is CORS and why is it needed?", category: "Web" },
  { question: "How do you secure an API?", category: "Web" },
  { question: "Explain the MVC architecture.", category: "Web" },
  { question: "What is the difference between supervised and unsupervised learning?", category: "ML" },
  { question: "Explain overfitting and underfitting.", category: "ML" },
  { question: "What is a confusion matrix?", category: "ML" },
  { question: "How does a decision tree work?", category: "ML" },
  { question: "What is cross-validation?", category: "ML" },
];

let currentIndex = 0;
let selectedQuestions = [];
let responses = [];

// DOM elements
const startBtn = document.getElementById("startBtn");
const questionBox = document.getElementById("questionBox");
const answerBox = document.getElementById("answerBox");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const buttons = document.getElementById("buttons");
const resultBox = document.getElementById("resultBox");
const feedbackContainer = document.getElementById("feedbackContainer");
const categoryForm = document.getElementById("categoryForm");

function getSelectedQuestions() {
  const selectedCategories = ["Behavioral", "Situational"];
  const checkboxes = categoryForm.querySelectorAll("input[type='checkbox']:checked");
  checkboxes.forEach(cb => selectedCategories.push(cb.value));

  const filtered = allQuestions.filter(q => selectedCategories.includes(q.category));

  if (filtered.length < 15) {
    alert(`Only ${filtered.length} questions found. Please select more domains.`);
    return [];
  }

  return filtered.sort(() => 0.5 - Math.random()).slice(0, 15);
}

function displayQuestion() {
  const q = selectedQuestions[currentIndex];
  if (!q) return;
  questionBox.textContent = `${currentIndex + 1}. ${q.question}`;
  answerBox.value = responses[currentIndex]?.answer || "";
  answerBox.style.display = "block";
  answerBox.focus();
}

function saveAnswer() {
  responses[currentIndex] = {
    question: selectedQuestions[currentIndex].question,
    answer: answerBox.value.trim()
  };
}

function showResults() {
  saveAnswer();
  questionBox.textContent = "You have completed the interview.";
  answerBox.style.display = "none";
  buttons.style.display = "none";
  resultBox.textContent = "Processing your responses...";

  fetch("http://localhost:8000/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ responses })
  })
  .then(res => res.json())
  .then(data => {
    resultBox.textContent = `Overall Score: ${data.overall_score}`;
    feedbackContainer.innerHTML = data.per_question_feedback.map(item => `
      <div style="margin-bottom:10px;">
        <strong>${item.question}</strong><br>
        <em>Matched Keywords:</em> ${item.matched_keywords.join(", ")}<br>
        <em>Keyword Match Score:</em> ${item.keyword_match_score}%<br>
        <em>Sentiment Score:</em> ${item.sentiment}<br>
        <em>Relevance Score:</em> ${item.relevance_score}<br>
      </div>
    `).join("");
  })
  .catch(err => {
    resultBox.textContent = "Failed to process answers.";
    console.error(err);
  });
}

startBtn.addEventListener("click", () => {
  selectedQuestions = getSelectedQuestions();
  if (selectedQuestions.length === 0) return;

  currentIndex = 0;
  responses = [];
  categoryForm.style.display = "none";
  buttons.style.display = "block";
  displayQuestion();
});

prevBtn.addEventListener("click", () => {
  saveAnswer();
  if (currentIndex > 0) {
    currentIndex--;
    displayQuestion();
  }
});

nextBtn.addEventListener("click", () => {
  saveAnswer();
  if (currentIndex < selectedQuestions.length - 1) {
    currentIndex++;
    displayQuestion();
  } else {
    showResults();
  }
});
