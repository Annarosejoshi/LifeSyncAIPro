(() => {
  const panel = document.getElementById('wellnessChatPanel');
  const form = document.getElementById('wellnessChatForm');
  if (!panel || !form) return;
  const input = document.getElementById('wellnessChatQuestion');
  const messages = document.getElementById('wellnessChatMessages');
  const launcher = document.querySelector('.chat-launcher');
  const send = form.querySelector('button[type="submit"]');
  let previousFocus;
  function open() {
    previousFocus = document.activeElement;
    panel.hidden = false;
    launcher.setAttribute('aria-expanded', 'true');
    input.focus();
  }
  function close() {
    panel.hidden = true;
    launcher.setAttribute('aria-expanded', 'false');
    (previousFocus || launcher).focus();
  }
  document.querySelectorAll('[data-open-chat]').forEach(button => button.addEventListener('click', open));
  document.getElementById('closeWellnessChat').addEventListener('click', close);
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !panel.hidden) close(); });
  document.querySelectorAll('[data-chat-question]').forEach(button => button.addEventListener('click', () => {
    input.value = button.dataset.chatQuestion;
    input.focus();
  }));
  function bubble(text, role) {
    const node = document.createElement('p');
    node.className = `chat-bubble ${role}`;
    node.textContent = text;
    messages.appendChild(node);
    messages.scrollTop = messages.scrollHeight;
    return node;
  }
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || send.disabled) return;
    bubble(question, 'user');
    input.value = '';
    send.disabled = true;
    const pending = bubble('Thinking…', 'assistant');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(form.action, {
        method: 'POST', credentials: 'same-origin', signal: controller.signal,
        headers: {'X-CSRFToken': form.elements.csrfmiddlewaretoken.value},
        body: new URLSearchParams({question})
      });
      if (response.redirected) throw new Error('Please sign in again to continue chatting.');
      if (!(response.headers.get('content-type') || '').includes('application/json')) throw new Error('Unable to answer right now. Please try again.');
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Please try again.');
      pending.textContent = payload.answer;
    } catch (error) {
      pending.textContent = error.name === 'AbortError' ? 'The reply took too long. Please try again.' : error.message;
      if (!input.value) input.value = question;
    } finally {
      clearTimeout(timeout);
      send.disabled = false;
      messages.scrollTop = messages.scrollHeight;
      if (!panel.hidden) input.focus();
    }
  });
})();
