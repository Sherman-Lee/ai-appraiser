let uploadedImages = [];
let progressPollingActive = false;
let currentValuationData = null;

const RESULTS_EMPTY_HTML =
  '<p id="results-empty-message" class="text-gray-500 text-sm">Your estimates will appear here after you add photos and click Get estimates.</p>';

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

function setSubmitError(message) {
  const el = document.getElementById("submit-error");
  if (!el) return;
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

function updateEstimateButtonState() {
  const btn = document.getElementById("estimate-value-button");
  if (btn) btn.disabled = uploadedImages.length === 0;
}

function updateExportButtonVisibility() {
  const btn = document.getElementById("download-csv-button");
  if (btn) {
    if (currentValuationData && currentValuationData.length > 0) {
      btn.classList.remove("hidden");
    } else {
      btn.classList.add("hidden");
    }
  }
}

function escapeCSVField(field) {
  if (field === null || field === undefined) {
    return "";
  }
  const str = String(field);
  // If field contains comma, quote, or newline, wrap in quotes and escape quotes
  if (str.includes(",") || str.includes('"') || str.includes("\n") || str.includes("\r")) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}

function exportToCSV() {
  if (!currentValuationData || currentValuationData.length === 0) {
    setSubmitError("No valuation results available to export.");
    return;
  }

  // Flatten all ValuationResponse objects from all images
  const rows = [];
  currentValuationData.forEach((entry) => {
    const imageIndex =
      entry && typeof entry.image_index === "number" ? entry.image_index : 0;
    const valuations = Array.isArray(entry.valuations)
      ? entry.valuations
      : entry.valuation
        ? [entry.valuation]
        : [];

    valuations.forEach((valuation) => {
      // Format search_urls array as semicolon-separated string
      const searchUrlsStr =
        valuation.search_urls && Array.isArray(valuation.search_urls)
          ? valuation.search_urls
              .filter((url) => url && url !== "N/A")
              .join("; ")
          : "";

      rows.push({
        image_index: imageIndex,
        item_name: valuation.item_name || "",
        estimated_value: valuation.estimated_value || 0,
        currency: valuation.currency || "",
        reasoning: valuation.reasoning || "",
        search_urls: searchUrlsStr,
      });
    });
  });

  if (rows.length === 0) {
    setSubmitError("No valuation data to export.");
    return;
  }

  // Create CSV header
  const headers = [
    "image_index",
    "item_name",
    "estimated_value",
    "currency",
    "reasoning",
    "search_urls",
  ];

  // Build CSV content
  const csvRows = [headers.map(escapeCSVField).join(",")];

  rows.forEach((row) => {
    const csvRow = headers.map((header) => escapeCSVField(row[header]));
    csvRows.push(csvRow.join(","));
  });

  const csvContent = csvRows.join("\n");

  // Create blob with UTF-8 BOM for Excel compatibility
  const BOM = "\uFEFF";
  const blob = new Blob([BOM + csvContent], { type: "text/csv;charset=utf-8;" });

  // Create download link and trigger download
  const link = document.createElement("a");
  const url = URL.createObjectURL(blob);
  link.setAttribute("href", url);

  // Generate filename with timestamp
  const now = new Date();
  const timestamp = now
    .toISOString()
    .replace(/[:.]/g, "-")
    .slice(0, -5);
  link.setAttribute("download", `valuations_${timestamp}.csv`);

  link.style.visibility = "hidden";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  // Clear any previous errors
  setSubmitError(null);
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
  const addMore = document.getElementById("add-more-photos");
  const hidden = document.getElementById("image-hidden-fields");

  grid.innerHTML = "";
  hidden.innerHTML = "";
  setUploadError(null);
  setSubmitError(null);
  updateEstimateButtonState();

  if (uploadedImages.length === 0) {
    if (empty) empty.classList.remove("hidden");
    if (addMore) addMore.classList.add("hidden");
    return;
  }

  if (empty) empty.classList.add("hidden");
  if (addMore) addMore.classList.remove("hidden");

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

// Generate a unique task ID function
function generateTaskId() {
  return 'task_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

document
  .getElementById("valuation-form")
  .addEventListener("htmx:beforeRequest", function (evt) {
    // Generate task ID on client side before request
    const taskId = generateTaskId();
    
    // Add task_id as a hidden form field
    const form = evt.target;
    let taskIdInput = form.querySelector('input[name="task_id"]');
    if (!taskIdInput) {
      taskIdInput = createHiddenInput("task_id", taskId);
      form.appendChild(taskIdInput);
    } else {
      taskIdInput.value = taskId;
    }

    // Function to start polling progress
    function startProgressPolling(taskId) {
      const progressContainer = document.getElementById("progress-container");
      const spinner = document.getElementById("spinner");
      
      // Show progress container and spinner
      progressContainer.classList.remove("hidden");
      spinner.classList.remove("hidden");
      
      // Initialize progress display
      const totalImages = uploadedImages.length;
      progressContainer.textContent = `0/${totalImages} images appraised`;

      const poll = () => {
        // If polling has been deactivated (e.g., request finished), stop immediately
        if (!progressPollingActive) return;

        fetch(`/progress/${taskId}`)
          .then(response => {
            if (!response.ok) {
              // Task might not exist yet, keep polling
              if (response.status === 404) {
                if (!progressPollingActive) return;
                setTimeout(poll, 500);
                return null;
              }
              throw new Error("Network response was not ok");
            }
            return response.json();
          })
          .then(data => {
            if (!data) return; // 404 case, already scheduled next poll
            
            // Update progress message
            const completed = data.completed || 0;
            const total = data.total || 0;
            progressContainer.textContent = `${completed}/${total} images appraised`;

            // Check if task is complete
            if (data.status === "completed") {
              // Stop polling; afterRequest will hide spinner
              progressPollingActive = false;
              return;
            }

            // Poll again after a short delay
            if (!progressPollingActive) return;
            setTimeout(poll, 500);
          })
          .catch(error => {
            console.error("Error fetching progress:", error);
            // Don't stop polling on error, just log it
            if (!progressPollingActive) return;
            setTimeout(poll, 1000);
          });
      };

      // Mark polling as active and start immediately
      progressPollingActive = true;
      poll();
    }

    // Start polling with the generated task ID
    startProgressPolling(taskId);
  });

document
  .getElementById("valuation-form")
  .addEventListener("htmx:afterRequest", function (evt) {
    // Stop any ongoing progress polling
    progressPollingActive = false;

    // Hide spinner and progress container
    const spinner = document.getElementById("spinner");
    const progressContainer = document.getElementById("progress-container");
    if (spinner) spinner.classList.add("hidden");
    if (progressContainer) progressContainer.classList.add("hidden");
    
    if (evt.detail.successful) {
      let payload;
      try {
        payload = JSON.parse(evt.detail.xhr.response);
      } catch (e) {
        // If we can't parse JSON, show a generic error and bail
        setSubmitError("Received an unexpected response from the server while getting appraisal results.");
        currentValuationData = null;
        updateExportButtonVisibility();
        return;
      }

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

      // Store current valuation data for CSV export
      currentValuationData = results;
      updateExportButtonVisibility();

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
          <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
            <div class="flex gap-4">
              <div class="w-24 h-24 shrink-0 border border-gray-200 rounded-lg bg-white overflow-hidden flex items-center justify-center">
                ${
                  thumb
                    ? `<img src="${thumb}" alt="Image ${imageIndex + 1}" class="w-full h-full object-contain" />`
                    : `<div class="text-xs text-gray-400 px-2 text-center">No preview</div>`
                }
              </div>
              <div class="min-w-0 flex-1 break-words">
                <p class="font-semibold text-gray-800">Image ${imageIndex + 1}</p>`;

        valuations.forEach((data, vIdx) => {
          const formattedValue = formatCurrency(
            data.estimated_value,
            data.currency
          );
          const rawName =
            data.item_name != null ? String(data.item_name).trim() : "";
          const itemLabel =
            rawName !== ""
              ? rawName
              : valuations.length > 1
                ? `Item ${vIdx + 1}`
                : "";
          resultsHTML += `
                <div class="mt-3 ${vIdx > 0 ? "pt-3 border-t border-gray-200" : ""}">
                  <p class="font-medium text-gray-700 mt-3 text-sm">${itemLabel}</p>
                  <p class="text-xs text-gray-500 mt-2">Estimated value</p>
                  <p class="text-3xl font-semibold text-primary-600 mt-0.5">${formattedValue}</p>
                  <p class="text-xs text-gray-500 mt-2">How we estimated</p>
                  <p class="text-sm text-gray-600">${data.reasoning}</p>`;

          if (
            data.search_urls &&
            data.search_urls.length > 0 &&
            data.search_urls[0] !== "N/A"
          ) {
            resultsHTML += `
                  <div class="mt-3">
                    <p class="text-xs text-gray-500 mt-2">Reference links</p>
                    <ul class="text-sm list-disc list-inside ml-4 mt-1">`;
            data.search_urls.forEach((url) => {
              resultsHTML += `<li class="break-all"><a href="${url}" target="_blank" rel="noopener noreferrer" class="text-primary-600 hover:underline break-all">${url}</a></li>`;
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
      // Clear any previous submit error on success
      setSubmitError(null);
      updateExportButtonVisibility();
    } else {
      // Handle failed request: surface error to the user if possible
      let message = "An error occurred while getting appraisal results.";
      try {
        const body = JSON.parse(evt.detail.xhr.response);
        if (body && body.detail) {
          message = body.detail;
        }
      } catch {
        // ignore JSON parse errors and keep generic message
      }
      setSubmitError(message);
      currentValuationData = null;
      updateExportButtonVisibility();
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

async function processFiles(files, input) {
  if (!files || files.length === 0) return;
  setUploadError(null);
  if (input) input.disabled = true;
  try {
    // Filter image files and create upload promises for parallel execution
    const imageFiles = Array.from(files).filter(file => file.type.startsWith("image/"));
    
    if (imageFiles.length === 0) {
      return;
    }

    // Upload all images in parallel
    const uploadPromises = imageFiles.map(file => 
      uploadOneFile(file).catch(error => {
        // Return error info instead of throwing to allow partial success
        return { error: error instanceof Error ? error.message : String(error), file: file.name };
      })
    );
    
    const results = await Promise.all(uploadPromises);
    
    // Process successful uploads and collect errors
    const errors = [];
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      if (result.error) {
        errors.push(`${result.file}: ${result.error}`);
      } else if (result && result.data_url) {
        uploadedImages.push({
          dataUrl: result.data_url,
          gcsUri: result.gcs_uri,
          contentType: result.content_type,
        });
      }
    }
    
    // Render once after all uploads complete
    if (uploadedImages.length > 0) {
      renderUploadedImages();
    }
    
    // Show errors if any occurred (but allow partial success)
    if (errors.length > 0) {
      const errorMsg = errors.length === imageFiles.length
        ? `Failed to upload images: ${errors.join("; ")}`
        : `Some images failed to upload: ${errors.join("; ")}`;
      setUploadError(errorMsg);
    }
  } catch (e) {
    setUploadError(e instanceof Error ? e.message : String(e));
  } finally {
    if (input) {
      input.value = "";
      input.disabled = false;
    }
  }
}

const imageFileInput = document.getElementById("image_file");
const imageFileAddInput = document.getElementById("image_file_add");
const emptyDropzone = document.getElementById("image-preview-empty");

if (emptyDropzone && imageFileInput) {
  emptyDropzone.addEventListener("click", function (e) {
    if (e.target === imageFileInput) return;
    imageFileInput.click();
  });
  emptyDropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      imageFileInput.click();
    }
  });
  emptyDropzone.addEventListener("dragover", function (e) {
    e.preventDefault();
    e.stopPropagation();
    emptyDropzone.classList.add("border-primary-600", "bg-gray-50");
  });
  emptyDropzone.addEventListener("dragleave", function (e) {
    e.preventDefault();
    e.stopPropagation();
    emptyDropzone.classList.remove("border-primary-600", "bg-gray-50");
  });
  emptyDropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    e.stopPropagation();
    emptyDropzone.classList.remove("border-primary-600", "bg-gray-50");
    const files = e.dataTransfer && e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
    processFiles(files, null);
  });
}

if (imageFileInput) {
  imageFileInput.addEventListener("change", async (evt) => {
    const input = evt.target;
    if (!(input instanceof HTMLInputElement)) return;
    await processFiles(Array.from(input.files || []), input);
  });
}

if (imageFileAddInput) {
  imageFileAddInput.addEventListener("change", async (evt) => {
    const input = evt.target;
    if (!(input instanceof HTMLInputElement)) return;
    await processFiles(Array.from(input.files || []), input);
  });
}

const estimateValueButton = document.getElementById(
  "estimate-value-button"
);
if (estimateValueButton) {
  estimateValueButton.addEventListener("click", function (event) {
    if (uploadedImages.length === 0) {
      setSubmitError("Please add at least one photo first.");
      event.preventDefault();
    } else {
      setSubmitError(null);
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

    // Clear valuation results and show empty state
    const resultsEl = document.getElementById("results");
    if (resultsEl) resultsEl.innerHTML = RESULTS_EMPTY_HTML;

    // Reset the currency selection
    document.getElementById("currency").value = defaultCurrency;

    // Clear valuation data and hide export button
    currentValuationData = null;
    updateExportButtonVisibility();
  });
}

const downloadCsvButton = document.getElementById("download-csv-button");
if (downloadCsvButton) {
  downloadCsvButton.addEventListener("click", function () {
    exportToCSV();
  });
}

// Initialize UI
renderUploadedImages();
updateEstimateButtonState();
updateExportButtonVisibility();