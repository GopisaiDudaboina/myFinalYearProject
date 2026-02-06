function setActiveTab(domain) {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    if (tab.dataset.domain === domain) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  const modePill = document.getElementById("modePill");
  if (modePill) {
    const label = domain === "telecom" ? "Telecom" : domain === "bank" ? "Banking" : "E-commerce";
    modePill.textContent = `${label} Mode`;
  }
}

async function updateFields() {
  const domainInput = document.getElementById("domain");
  const domain = domainInput.value;
  const container = document.getElementById("inputFields");

  container.innerHTML = "<div class=\"empty\">Loading fields...</div>";
  container.classList.add("loading");

  try {
    const optRes = await fetch(`/feature-options/${domain}`);
    const options = await optRes.json();

    const numRes = await fetch(`/features/${domain}`);
    const numeric = await numRes.json();

    const fragments = [];

    Object.keys(options).forEach((feature) => {
      const values = options[feature];
      let html = `<div class="field"><label>${feature}</label><select name="${feature}" required>`;
      values.forEach((val) => {
        html += `<option value="${val}">${val}</option>`;
      });
      html += `</select></div>`;
      fragments.push(html);
    });

    numeric.forEach((feature) => {
      fragments.push(`
        <div class="field">
          <label>${feature}</label>
          <input type="number" step="any" name="${feature}" placeholder="${feature}" required>
        </div>
      `);
    });

    container.innerHTML = fragments.join("");
    container.classList.remove("loading");
  } catch (error) {
    container.innerHTML = "<div class=\"empty\">Error loading fields</div>";
    container.classList.remove("loading");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const domainInput = document.getElementById("domain");
  const tabs = document.querySelectorAll(".tab");
  const resetBtn = document.getElementById("resetBtn");
  const form = document.querySelector("form");
  const resultsSection = document.getElementById("resultsSection");

  if (!domainInput || !form) {
    return;
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const domain = tab.dataset.domain;
      domainInput.value = domain;
      setActiveTab(domain);
      updateFields();
    });
  });

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      updateFields();
    });
  }

  if (form) {
    form.addEventListener("submit", () => {
      if (resultsSection) {
        resultsSection.classList.add("loading");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

  setActiveTab(domainInput.value);
  updateFields();
});
