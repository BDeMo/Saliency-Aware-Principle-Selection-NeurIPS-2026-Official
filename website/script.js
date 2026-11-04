const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.14 });

document.querySelectorAll('.reveal, .chart-card').forEach((el) => revealObserver.observe(el));

document.querySelectorAll('.bars').forEach((bars) => {
  [...bars.children].forEach((bar, index) => bar.style.setProperty('--order', index));
});

const progressBar = document.querySelector('#scroll-progress-bar');
const updateProgress = () => {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
  progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
};
window.addEventListener('scroll', updateProgress, { passive: true });
updateProgress();

const citation = `@inproceedings{shi2026sap,
  title={Saliency-Aware Multi-Route Thinking: Grounding and Reasoning on Vision-Language Agents},
  author={Shi, Mingjia and He, Yinhan and Zhu, Yaochen and Dong, Cassie and Li, Jundong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}`;

document.querySelector('#cite-button').addEventListener('click', async (event) => {
  await navigator.clipboard.writeText(citation);
  const button = event.currentTarget;
  const previous = button.textContent;
  button.textContent = 'Copied';
  setTimeout(() => { button.textContent = previous; }, 1600);
});
