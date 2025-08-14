document.addEventListener('DOMContentLoaded', () => {
    const askButton = document.getElementById('ask-button');
    const questionInput = document.getElementById('question-input');
    const answerOutput = document.getElementById('answer-output');
    const loader = document.getElementById('loader');

    askButton.addEventListener('click', async () => {
        const question = questionInput.value;
        if (!question.trim()) {
            alert('Please enter a question.');
            return;
        }

        // Show loader and clear previous answer
        loader.style.display = 'block';
        answerOutput.innerHTML = '';
        askButton.disabled = true;

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            // Display the answer from the backend
            answerOutput.innerHTML = `<p>${data.answer.replace(/\n/g, '<br>')}</p>`;

        } catch (error) {
            console.error('Error:', error);
            answerOutput.innerHTML = '<p style="color: red;">Failed to get an answer. Please check the console for errors.</p>';
        } finally {
            // Hide loader and re-enable button
            loader.style.display = 'none';
            askButton.disabled = false;
        }
    });
});