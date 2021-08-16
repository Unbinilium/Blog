async function setSidebarWrapper() {
    if (window.innerWidth >= 900) {
        const sidebarMaxHeight = window.innerHeight - 60 + "px";
        document.documentElement.style.setProperty("--sidebar-wrapper-height", sidebarMaxHeight);
    } else {
        document.documentElement.style.setProperty("--sidebar-wrapper-height", "max-content");
    }
}

setSidebarWrapper();

window.addEventListener('resize', setSidebarWrapper);
