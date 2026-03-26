(() => {
    const navToggle = document.querySelector('[data-nav-toggle]');
    const nav = document.querySelector('[data-nav]');

    function setNavOpen(isOpen) {
        if (!navToggle || !nav) {
            return;
        }
        nav.classList.toggle('is-open', isOpen);
        navToggle.setAttribute('aria-expanded', String(isOpen));
    }

    if (navToggle && nav) {
        navToggle.addEventListener('click', () => {
            const isOpen = navToggle.getAttribute('aria-expanded') === 'true';
            setNavOpen(!isOpen);
        });

        nav.addEventListener('click', (event) => {
            const target = event.target;
            if (target instanceof HTMLElement && target.closest('a, button')) {
                setNavOpen(false);
            }
        });

        window.addEventListener('resize', () => {
            if (window.innerWidth > 760) {
                setNavOpen(false);
            }
        });

        window.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                setNavOpen(false);
            }
        });
    }

    document.addEventListener('click', (event) => {
        if (
            navToggle &&
            nav &&
            nav.classList.contains('is-open') &&
            event.target instanceof HTMLElement &&
            !event.target.closest('[data-nav]') &&
            !event.target.closest('[data-nav-toggle]')
        ) {
            setNavOpen(false);
        }

        const trigger = event.target instanceof HTMLElement ? event.target.closest('[data-confirm]') : null;
        if (!trigger) {
            return;
        }

        const message = trigger.getAttribute('data-confirm');
        if (message && !window.confirm(message)) {
            event.preventDefault();
            event.stopPropagation();
        }
    });
})();
