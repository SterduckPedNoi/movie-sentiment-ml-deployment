document.addEventListener("DOMContentLoaded", () => {
document.addEventListener("click", (e) => {
  if (!movieInput.contains(e.target) &&
      !suggestionBox.contains(e.target)) {
    suggestionBox.classList.add("hidden");
  }
});

  // ================= CONFIG =================
  const API_BASE = window.location.origin;
  const API_COMPARE = API_BASE + "/compare";

  let tags = [];
  
  const GENRE_MAP = {
  28: "🎬 Action",
  12: "🧭 Adventure",
  16: "🎨 Animation",
  35: "😂 Comedy",
  80: "🚓 Crime",
  99: "📚 Documentary",
  18: "🎭 Drama",
  10751: "👨‍👩‍👧 Family",
  14: "🧙 Fantasy",
  36: "🏛️ History",
  27: "😱 Horror",
  10402: "🎶 Musical",
  9648: "🕵️ Mystery",
  10749: "❤️ Romance",
  878: "🚀 Science Fiction",
  53: "🔪 Thriller",
  10752: "⚔️ War",
  37: "🤠 Western"
};

  // ================= ELEMENTS =================
  
  const modelA = document.getElementById("modelA");
  const modelB = document.getElementById("modelB");
  const addTagBtn = document.getElementById("addTagBtn");
  const tagSelect = document.getElementById("tagSelect");
  const tagContainer = document.getElementById("tagContainer");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const reviewText = document.getElementById("reviewText");
  const loadingSection = document.getElementById("loadingSection");
  const movieInput = document.getElementById("movieName");
  const suggestionBox = document.getElementById("movieSuggestions");
  const result = document.getElementById("result");
  const resultSection = document.getElementById("resultSection");
  const movieTitleShow = document.getElementById("movieTitleShow");
  const tagShow = document.getElementById("tagShow");

  console.log("movieInput:", movieInput);
  console.log("suggestionBox:", suggestionBox);

  // ================= LOAD MODELS =================
  async function loadModels() {
    const res = await fetch(API_BASE + "/models");
    const models = await res.json();

    modelA.innerHTML = "";
    modelB.innerHTML = "";

    models.forEach(m => {
      const label = m === "v5_ensemble" ? `${m} 🔥` : m;
      modelA.innerHTML += `<option value="${m}">${label}</option>`;
      modelB.innerHTML += `<option value="${m}">${label}</option>`;
    });

    modelA.value = models[0];
    modelB.value = models[1] || models[0];
  }

  loadModels();


/* ================= TAG SYSTEM ================= */
addTagBtn.onclick = () => {
  const t = tagSelect.value;
  if (!t || tags.includes(t)) return;
  tags.push(t);
  renderTags();
};

function renderTags() {
  tagContainer.innerHTML = "";
  tags.forEach(t => {
    tagContainer.innerHTML += `
      <span onclick="removeTag('${t}')"
        class="bg-gradient-to-r from-blue-500/20 to-purple-500/20
               border border-white/10 text-blue-300
               px-4 py-1 rounded-full text-sm cursor-pointer
               hover:scale-105 transition">
        ${t} ✕
      </span>`;
  });
}

window.removeTag = function (t) {
  tags = tags.filter(x => x !== t);
  renderTags();
};

/* ================= ANALYZE ================= */
analyzeBtn.onclick = async () => {
  if (!reviewText.value.trim()) {
    alert("Please write a review. 😅");
    return;
  }

  // ===== START LOADING =====
  analyzeBtn.disabled = true;
  analyzeBtn.innerText = "⏳ Analyzing...";
  loadingSection.classList.remove("hidden");
  resultSection.classList.add("hidden");

  try {
    const res = await fetch(API_COMPARE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: reviewText.value,
        model_a: modelA.value,
        model_b: modelB.value,
        tags: tags,
        movie_name: movieInput.value
      })
    });

    const data = await res.json();
    renderResult(data);

  } catch (err) {
    alert("❌ Analysis failed. Please try again.
");
    console.error(err);
  }

  // ===== END LOADING =====
  analyzeBtn.disabled = false;
  analyzeBtn.innerText = "🚀 Analyze Review";
  loadingSection.classList.add("hidden");
};
function renderResult(data) {
  result.innerHTML = "";
  resultSection.classList.remove("hidden");

  movieTitleShow.innerText =
    "🎞️ " + (data.movie_name || "ไม่ระบุชื่อหนัง");

  tagShow.innerHTML = data.tags.map(t =>
    `<span class="bg-white/10 px-3 py-1 rounded-full text-sm">${t}</span>`
  ).join("");

  for (const [m, r] of Object.entries(data.results)) {
    const positive = r.sentiment === "Positive";
    const color = positive
      ? "from-emerald-400 to-green-500"
      : "from-rose-400 to-red-500";

    const w = Math.round(r.confidence * 100);

    result.innerHTML += `
      <div class="bg-white/5 backdrop-blur-xl border border-white/10
                  p-6 rounded-2xl hover:scale-[1.02] transition">

        <h3 class="font-semibold text-lg mb-1">🤖 ${m}</h3>

        <p class="text-xl font-extrabold
           bg-gradient-to-r ${color}
           text-transparent bg-clip-text">
          ${r.sentiment}
        </p>

        <p class="text-sm mt-2">Confidence: ${w}%</p>

        <div class="w-full h-2 bg-white/10 rounded-full overflow-hidden mt-2">
          <div class="h-full bg-gradient-to-r ${color}"
               style="width:${w}%"></div>
        </div>

        <p class="text-sm mt-4">⏱ Latency: ${r.latency_ms} ms</p>

        <p class="text-sm mt-3">🔑 Keywords</p>
        <div class="flex gap-2 mt-2 flex-wrap">
          ${r.keywords.map(k =>
            `<span class="bg-white/10 px-2 py-1 rounded text-sm">${k}</span>`
          ).join("")}
        </div>
      </div>
    `;
  }
}

/* ================= MOVIE AUTOCOMPLETE ================= */

let debounceTimer = null;

movieInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);

  const q = movieInput.value.trim();
  if (q.length < 2) {
    suggestionBox.classList.add("hidden");
    return;
  }

  debounceTimer = setTimeout(async () => {
    try {
      const res = await fetch(`${API_BASE}/search_movie?q=${q}`);
      const movies = await res.json();

      suggestionBox.innerHTML = "";

      if (!movies || movies.length === 0) {
        suggestionBox.classList.add("hidden");
        return;
      }

      movies.forEach(m => {
        const div = document.createElement("div");
        div.className =
          "px-4 py-2 cursor-pointer hover:bg-purple-500/20 transition";
        div.innerText = `${m.title} (${m.year || "N/A"})`;

        div.onclick = () => {
          movieInput.value = `${m.title} (${m.year || "N/A"})`;
          suggestionBox.classList.add("hidden");

          // 🔥 link genre -> tag อัตโนมัติ
          if (m.genre_ids && Array.isArray(m.genre_ids)) {
            m.genre_ids.forEach(id => {
              const tag = GENRE_MAP[id];
              if (tag && !tags.includes(tag)) {
                tags.push(tag);
              }
            });
            renderTags();
          }
        };



        suggestionBox.appendChild(div);
      });

      suggestionBox.classList.remove("hidden");
    } catch (err) {
      console.error("Movie search error", err);
    }
  }, 300);

})

});

movieInput.addEventListener("input", () => {
  if (movieInput.value.length === 0) {
    tags = [];
    renderTags();
  }
});
