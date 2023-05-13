function scrollToElement(element, offset = 20, duration = 500) {
  const startingY = window.pageYOffset;
  const targetY = element.getBoundingClientRect().top + startingY - offset;

  const diff = targetY - startingY;
  let start;

  if (!diff) return;

  window.requestAnimationFrame(function step(timestamp) {
    if (!start) start = timestamp;
    const time = timestamp - start;
    const percent = Math.min(time / duration, 1);
    window.scrollTo(0, startingY + diff * percent);

    if (time < duration) {
      window.requestAnimationFrame(step);
    }
  });
}

document.addEventListener("DOMContentLoaded", function () {
  const tocLinks = document.querySelectorAll(".nav-item a");

  console.log(tocLinks);
  tocLinks.forEach(function (link) {
    link.addEventListener("click", function (event) {
      event.preventDefault();

      const currentActive = document.querySelector(
        ".bd-sidebar .nav-item.active"
      );
      if (currentActive) {
        currentActive.classList.remove("active");
      }

      // Find the closest .nav-item and add the 'active' class
      const navItem = link.closest(".nav-item");
      if (navItem) {
        navItem.classList.add("active");
      }

      // Add the 'active' class to the clicked link
    //   link.classList.add("active");

      const targetElementId = new URL(link.href).hash.substring(1);
      const targetElement = document.getElementById(targetElementId);

      if (targetElement) {
        scrollToElement(targetElement, 100, 500);
        setTimeout(() => {
          window.location.href = link.href;
        }, 500);
      } else {
        window.location.href = link.href;
      }
    });
  });
});
