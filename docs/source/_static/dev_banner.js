// Dismiss the development-version banner when its close (×) button is clicked.
// Uses event delegation so it works regardless of when the banner is rendered.
document.addEventListener("click", function (event) {
  var button = event.target.closest(".mlq-dev-banner__close");
  if (!button) {
    return;
  }
  var bar = button.closest(".bd-header-announcement") || button.closest(".mlq-dev-banner");
  if (bar) {
    bar.style.display = "none";
  }
});
