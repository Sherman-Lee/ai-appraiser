let uploadedImages = [];

function setUploadError(message) {
  const el = document.getElementById("upload-error");
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

function createHiddenInput(name, value) {
  const input = document.createElement("input");
  input.type = "hidden";
  input.name = name;
  input.value = value;
  return input;
}

function renderUploadedImages() {
  const empty = document.getElementById("image-preview-empty");
  const grid = document.getElementById("image-preview-grid");
  const hidden = document.getElementById("image-hidden-fields");

  grid.innerHTML = "";
  hidden.innerHTML = "";
  setUploadError(null);

  if (uploadedImages.length === 0) {
    empty.classList.remove("hidden");
    return;
  }

  empty.classList.add("hidden");

  uploadedImages.forEach((img, idx) => {
    const card = document.createElement("div");
    card.className =
      "relative border border-gray-300 rounded-lg bg-white overflow-hidden shadow-sm";
    card.innerHTML = `
        <img src="${img.dataUrl}" alt="Uploaded image ${idx + 1}" class="w-full h-28 object-contain bg-white" />
        <button type="button" class="absolute top-1 right-1 bg-white/90 hover:bg-white text-gray-700 border border-gray-300 rounded px-2 py-0.5 text-xs" data-remove-index="${idx}">
          Remove
        </button>
      `;
    grid.appendChild(card);
  });

  // Hidden fields: send ordered items so backend can value per image.
  uploadedImages.forEach((img) => {
    const item = img.gcsUri
      ? {
          kind: "gcs",
          gcs_uri: img.gcsUri,
          data_url: img.dataUrl,
          content_type: img.contentType || "image/jpeg",
        }
      : {
          kind: "inline",
          data_url: img.dataUrl,
          content_type: img.contentType || "image/jpeg",
        };
    hidden.appendChild(createHiddenInput("image_items", JSON.stringify(item)));
  });
}

document
  .getElementById("valuation-form")
  .addEventListener("htmx:beforeRequest", function () {
    document.getElementById("spinner").classList.remove("hidden");
  });

document
  .getElementById("valuation-form")
  .addEventListener("htmx:afterRequest", function (evt) {
    document.getElementById("spinner").classList.add("hidden");
    if (evt.detail.successful) {
      const payload = JSON.parse(evt.detail.xhr.response);

      // Format the currency
      function formatCurrency(value, currency) {
        return new Intl.NumberFormat("en-US", {
          style: "currency",
          currency: currency,
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }).format(value);
      }

      let results = payload && payload.results;
      if (!Array.isArray(results)) {
        // Back-compat: if server returns a single ValuationResponse object.
        results = [{ image_index: 0, valuation: payload }];
      }

      let resultsHTML = `<div class="space-y-4">`;

      results.forEach((entry) => {
        const imageIndex =
          entry && typeof entry.image_index === "number"
            ? entry.image_index
            : 0;
        const valuations = Array.isArray(entry.valuations)
          ? entry.valuations
          : entry.valuation
            ? [entry.valuation]
            : [];

        const thumb = uploadedImages[imageIndex]
          ? uploadedImages[imageIndex].dataUrl
          : null;

        resultsHTML += `
          <div class="bg-white p-4 rounded-lg shadow-md">
            <div class="flex gap-4">
              <div class="w-24 h-24 shrink-0 border border-gray-200 rounded bg-white overflow-hidden flex items-center justify-center">
                ${
                  thumb
                    ? `<img src="${thumb}" alt="Image ${imageIndex + 1}" class="w-full h-full object-contain" />`
                    : `<div class="text-xs text-gray-400 px-2 text-center">No preview</div>`
                }
              </div>
              <div class="min-w-0 flex-1">
                <p class="font-semibold">Image ${imageIndex + 1}</p>`;

        valuations.forEach((data, vIdx) => {
          const formattedValue = formatCurrency(
            data.estimated_value,
            data.currency
          );
          const itemLabel =
            valuations.length > 1 ? `Item ${vIdx + 1}: ` : "";
          resultsHTML += `
                <div class="mt-3 ${vIdx > 0 ? "pt-3 border-t border-gray-200" : ""}">
                  <p class="font-semibold mt-2">${itemLabel}Estimated Value:</p>
                  <p class="text-2xl text-blue-600">${formattedValue}</p>
                  <p class="font-semibold mt-3">Reasoning:</p>
                  <p class="text-sm text-gray-700">${data.reasoning}</p>`;

          if (
            data.search_urls &&
            data.search_urls.length > 0 &&
            data.search_urls[0] !== "N/A"
          ) {
            resultsHTML += `
                  <div class="mt-3">
                    <p class="font-semibold">Sources:</p>
                    <ul class="text-sm list-disc list-inside ml-4">`;
            data.search_urls.forEach((url) => {
              resultsHTML += `<li><a href="${url}" target="_blank" class="text-blue-600 hover:underline">${url}</a></li>`;
            });
            resultsHTML += `</ul></div>`;
          }
          resultsHTML += `</div>`;
        });

        resultsHTML += `
              </div>
            </div>
          </div>`;
      });

      resultsHTML += `</div>`;
      document.getElementById("results").innerHTML = resultsHTML;
    }
  });

document
  .getElementById("image-preview-grid")
  .addEventListener("click", function (evt) {
    const target = evt.target;
    if (!(target instanceof HTMLElement)) return;
    const removeIndexAttr = target.getAttribute("data-remove-index");
    if (removeIndexAttr === null) return;
    const idx = Number(removeIndexAttr);
    if (!Number.isFinite(idx)) return;
    uploadedImages.splice(idx, 1);
    renderUploadedImages();
  });

async function uploadOneFile(file) {
  const formData = new FormData();
  formData.append("image_file", file);

  const resp = await fetch("/upload-image", {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    let detail = "An error occurred while uploading the image.";
    try {
      const body = await resp.json();
      if (body && body.detail) detail = body.detail;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail);
  }

  return await resp.json();
}

document.getElementById("image_file").addEventListener("change", async (evt) => {
  const input = evt.target;
  if (!(input instanceof HTMLInputElement)) return;
  const files = input.files ? Array.from(input.files) : [];
  if (files.length === 0) return;

  setUploadError(null);
  input.disabled = true;

  try {
    for (const file of files) {
      if (!file.type.startsWith("image/")) {
        continue;
      }
      const data = await uploadOneFile(file);
      if (data && data.data_url) {
        uploadedImages.push({
          dataUrl: data.data_url,
          gcsUri: data.gcs_uri,
          contentType: data.content_type,
        });
        renderUploadedImages();
      }
    }
  } catch (e) {
    setUploadError(e instanceof Error ? e.message : String(e));
  } finally {
    input.value = "";
    input.disabled = false;
  }
});

const estimateValueButton = document.getElementById(
  "estimate-value-button"
);
if (estimateValueButton) {
  estimateValueButton.addEventListener("click", function (event) {
    if (uploadedImages.length === 0) {
      alert("Please upload at least one image first.");
      event.preventDefault();
    }
  });
}

const resetButton = document.getElementById("reset-button");
if (resetButton) {
  resetButton.addEventListener("click", function () {
    // Clear the description field
    document.getElementById("description").value = "";

    // Reset uploaded images + hidden fields
    uploadedImages = [];
    renderUploadedImages();

    // Clear valuation results
    document.getElementById("results").innerHTML = "";

    // Reset the currency selection
    document.getElementById("currency").value = defaultCurrency;
  });
}

// Initialize UI
renderUploadedImages();