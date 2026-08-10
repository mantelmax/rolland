document.addEventListener("DOMContentLoaded", function () {
    const sidebar = document.getElementById("left-sidebar");
    if (!sidebar) return;

    const listItems = sidebar.querySelectorAll("li");
    listItems.forEach(function (item) {
        const subMenu = item.querySelector("ul");
        if (subMenu) {
            item.classList.add("has-dropdown");

            const link = item.querySelector(":scope > a");
            if (link) {
                const toggleBtn = document.createElement("button");
                toggleBtn.className = "sidebar-dropdown-toggle";
                toggleBtn.type = "button";
                toggleBtn.setAttribute("aria-label", "Toggle dropdown");
                toggleBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>';

                link.appendChild(toggleBtn);

                if (item.classList.contains("current") || item.querySelector("li.current")) {
                    item.classList.add("expanded");
                }

                toggleBtn.addEventListener("click", function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    item.classList.toggle("expanded");
                });
            }
        }
    });
});
