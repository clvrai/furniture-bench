document.addEventListener("DOMContentLoaded", function () {
  const sections = document.querySelectorAll("section");

  sections.forEach(function (section) {
    const lineBreak = document.createElement("br");
    section.appendChild(lineBreak);
  });
});
