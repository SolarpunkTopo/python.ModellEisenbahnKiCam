// static/app.js – Eisenbahn Koppelabstand

// ---------------- Status laden ----------------

async function fetchStatus() {
    try {
        const res = await fetch('/status');
        if (!res.ok) {
            throw new Error('HTTP ' + res.status);
        }
        return await res.json();
    } catch (e) {
        console.error('Status-Fetch Fehler:', e);
        return null;
    }
}

function applyStatusToHeader(data) {
    const dot = document.getElementById('headerStatusDot');
    const label = document.getElementById('headerStatusLabel');
    if (!dot || !label) return;

    function setState(state, text) {
        dot.classList.remove('ok', 'alarm', 'neutral');
        dot.classList.add(state);
        label.textContent = text;
    }

    if (!data || typeof data.coupling === 'undefined') {
        setState('neutral', 'Keine Daten');
        return;
    }

    if (data.coupling) {
        setState('alarm', 'ALARM');
    } else {
        setState('ok', 'OK');
    }
}

function applyStatusToPage(data) {
    const alarm = document.getElementById('alarm');
    const details = document.getElementById('details');

    function setBox(state, text) {
        if (alarm) {
            alarm.className = 'alarm-box ' + state;
            alarm.textContent = text;
        }
    }

    // Basiswerte aus /status
    const coupling = data && !!data.coupling;
    const minDist = data && typeof data.min_distance_px === 'number'
        ? data.min_distance_px
        : null;
    const threshold = data && typeof data.threshold_px === 'number'
        ? data.threshold_px
        : null;
    const lokCount = data && typeof data.lok_count === 'number'
        ? data.lok_count
        : 0;
    const wagonCount = data && typeof data.wagon_count === 'number'
        ? data.wagon_count
        : 0;
    const mqttInfo = data && data.mqtt ? data.mqtt : null;

    // ---- Statusbox & Details ----
    if (!data || typeof data.coupling === 'undefined') {
        setBox('neutral', 'Keine Daten');
        if (details) details.textContent = '';
    } else {
        if (coupling) {
            setBox('alarm', 'ALARM: WAGON AUF KOPPELABSTAND');
        } else {
            setBox('ok', 'OK: Kein Koppelabstand');
        }

        if (details) {
            let txt = `Loks: ${lokCount}, Waggons: ${wagonCount}`;
            if (minDist !== null) {
                txt += ` | minimaler Abstand: ${minDist.toFixed(1)} px`;
            } else {
                txt += ` | minimaler Abstand: n/a`;
            }
            if (threshold !== null) {
                txt += ` | Grenzwert: ${threshold.toFixed(1)} px`;
            }

            if (coupling && minDist !== null && threshold !== null) {
                txt += ` → KOPPELABSTAND AKTIV`;
            }

            if (mqttInfo && mqttInfo.topic && mqttInfo.payload && mqttInfo.timestamp) {
                txt += ` | MQTT: ${mqttInfo.topic} → ${mqttInfo.payload} @ ${mqttInfo.timestamp}`;
            }

            details.textContent = txt;
        }
    }

    // ---- Rahmen um alle Video-Container (/ und /monitor) ----
    const wrappers = document.querySelectorAll('.img-wrapper');
    wrappers.forEach(w => {
        w.classList.remove('alarm-border', 'ok-border');
        if (data && typeof data.coupling !== 'undefined') {
            if (coupling) {
                w.classList.add('alarm-border');
            } else {
                w.classList.add('ok-border');
            }
        }
    });

    // ---- Overlay-Label direkt im Videobild ----
    const overlayLabels = document.querySelectorAll('.video-overlay-label');
    overlayLabels.forEach(lbl => {
        lbl.classList.remove('alarm', 'ok', 'neutral');
        if (!data || typeof data.coupling === 'undefined') {
            lbl.classList.add('neutral');
            lbl.textContent = 'Keine Daten';
        } else if (coupling) {
            lbl.classList.add('alarm');
            lbl.textContent = 'WAGON AUF KOPPELABSTAND';
        } else {
            lbl.classList.add('ok');
            lbl.textContent = 'Kein Koppelabstand';
        }
    });

    // ---- Großer Status unter dem Video (nur /monitor) ----
    const monitorCount = document.getElementById('monitorCountLabel');
    if (monitorCount) {
        monitorCount.classList.remove('alarm', 'ok');
        if (!data || typeof data.coupling === 'undefined') {
            monitorCount.textContent = 'Keine Daten';
        } else if (coupling) {
            monitorCount.textContent = 'Koppelabstand erkannt';
            monitorCount.classList.add('alarm');
        } else {
            monitorCount.textContent = 'Kein Koppelabstand';
            monitorCount.classList.add('ok');
        }
    }

    // ---- Zusatzinfos (index & monitor) ----
    const minDistanceInfo = document.getElementById('minDistanceInfo');
    if (minDistanceInfo) {
        minDistanceInfo.textContent =
            minDist !== null ? `${minDist.toFixed(1)} px` : '–';
    }

    const thresholdInfo = document.getElementById('thresholdInfo');
    if (thresholdInfo) {
        thresholdInfo.textContent =
            threshold !== null ? `${threshold.toFixed(1)} px` : '–';
    }

    const lokCountInfo = document.getElementById('lokCountInfo');
    if (lokCountInfo) {
        lokCountInfo.textContent = lokCount;
    }

    const wagonCountInfo = document.getElementById('wagonCountInfo');
    if (wagonCountInfo) {
        wagonCountInfo.textContent = wagonCount;
    }

    const mqttInfoLabel = document.getElementById('mqttInfo');
    if (mqttInfoLabel) {
        if (mqttInfo && mqttInfo.topic && mqttInfo.payload && mqttInfo.timestamp) {
            mqttInfoLabel.textContent =
                `${mqttInfo.topic} → ${mqttInfo.payload} @ ${mqttInfo.timestamp}`;
        } else if (mqttInfo && mqttInfo.enabled === false) {
            mqttInfoLabel.textContent = 'MQTT deaktiviert';
        } else {
            mqttInfoLabel.textContent = 'Noch kein MQTT-Event';
        }
    }

    // ---- Bounding-Boxen im Monitor zeichnen (Lok/Waggon) ----
    const bboxLayer = document.getElementById('bboxLayer');
    if (bboxLayer) {
        while (bboxLayer.firstChild) {
            bboxLayer.removeChild(bboxLayer.firstChild);
        }

        const wrapper = bboxLayer.closest('.img-wrapper');
        const img = wrapper ? wrapper.querySelector('img') : null;

        if (img && img.naturalWidth && img.naturalHeight && data && data.detections) {
            const dispW = img.clientWidth || img.naturalWidth;
            const dispH = img.clientHeight || img.naturalHeight;
            const scaleX = dispW / img.naturalWidth;
            const scaleY = dispH / img.naturalHeight;

            data.detections.forEach(det => {
                const x1 = det.x1;
                const y1 = det.y1;
                const x2 = det.x2;
                const y2 = det.y2;

                const left = x1 * scaleX;
                const top = y1 * scaleY;
                const width = (x2 - x1) * scaleX;
                const height = (y2 - y1) * scaleY;

                const role = det.role || 'other';
                const label = (det.label && typeof det.label === 'string')
                    ? det.label
                    : (role === 'lok' ? 'Lok' : (role === 'wagon' ? 'Waggon' : 'Objekt'));

                const box = document.createElement('div');
                box.className = 'bbox-box';
                box.style.left = left + 'px';
                box.style.top = top + 'px';
                box.style.width = width + 'px';
                box.style.height = height + 'px';

                const lblDiv = document.createElement('div');
                lblDiv.className = 'bbox-label';
                lblDiv.textContent = label;

                // Farben nach Rolle
                if (role === 'lok') {
                    box.style.borderColor = '#00f';
                    lblDiv.style.backgroundColor = 'rgba(0, 0, 255, 0.6)';
                } else if (role === 'wagon') {
                    box.style.borderColor = '#0f0';
                    lblDiv.style.backgroundColor = 'rgba(0, 128, 0, 0.6)';
                } else {
                    box.style.borderColor = '#ccc';
                    lblDiv.style.backgroundColor = 'rgba(128, 128, 128, 0.6)';
                }

                box.appendChild(lblDiv);
                bboxLayer.appendChild(box);
            });
        }
    }

    // ---- Thumbnail-Grid rechts im Monitor: Ausschnitte aus dem Videobild ----
    const grid = document.getElementById('detThumbGrid');
    if (grid) {
        while (grid.firstChild) {
            grid.removeChild(grid.firstChild);
        }

        if (data && data.detections && data.detections.length > 0) {
            const videoWrapper = document.querySelector('.monitor-video .img-wrapper');
            const videoImg = videoWrapper ? videoWrapper.querySelector('img') : null;

            if (videoImg && videoImg.complete && videoImg.naturalWidth && videoImg.naturalHeight) {
                data.detections.forEach(det => {
                    const sx = det.x1;
                    const sy = det.y1;
                    const sw = det.x2 - det.x1;
                    const sh = det.y2 - det.y1;

                    if (sw <= 0 || sh <= 0) return;

                    const thumbDiv = document.createElement('div');
                    thumbDiv.className = 'det-thumb';

                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');

                    const baseSize = 120;
                    let cw, ch;
                    if (sw >= sh) {
                        cw = baseSize;
                        ch = Math.round(baseSize * (sh / sw));
                    } else {
                        ch = baseSize;
                        cw = Math.round(baseSize * (sw / sh));
                    }
                    canvas.width = cw;
                    canvas.height = ch;

                    try {
                        ctx.drawImage(
                            videoImg,
                            sx, sy, sw, sh,
                            0, 0, cw, ch
                        );
                    } catch (e) {
                        console.error('Fehler beim Zeichnen des Thumbnails:', e);
                    }

                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'det-thumb-label';
                    const role = det.role || 'other';
                    const lblText = (det.label && typeof det.label === 'string')
                        ? det.label
                        : (role === 'lok' ? 'Lok' : (role === 'wagon' ? 'Waggon' : 'Objekt'));
                    labelDiv.textContent = lblText;

                    thumbDiv.appendChild(canvas);
                    thumbDiv.appendChild(labelDiv);
                    grid.appendChild(thumbDiv);
                });
            }
        }
    }
}

async function updateStatus() {
    const data = await fetchStatus();
    applyStatusToHeader(data);
    applyStatusToPage(data);
}

// ---------------- Config-Handling (/index) ----------------

async function loadConfig() {
    const rtspInput     = document.getElementById('configRtspUrl');
    const procInput     = document.getElementById('configProcessInterval');
    const confInput     = document.getElementById('configConfThresh');
    const imgInput      = document.getElementById('configImgSz');
    const fwInput       = document.getElementById('configFrameWidth');
    const fhInput       = document.getElementById('configFrameHeight');
    const couplingInput = document.getElementById('configCouplingDistancePx');

    const mqttEnabled = document.getElementById('configMqttEnabled');
    const mqttHost    = document.getElementById('configMqttHost');
    const mqttPort    = document.getElementById('configMqttPort');
    const mqttTopic   = document.getElementById('configMqttTopic');
    const mqttPayload = document.getElementById('configMqttPayloadOn');
    const mqttUser    = document.getElementById('configMqttUser');
    const mqttPass    = document.getElementById('configMqttPass');

    // wenn keine dieser Controls existiert, sind wir z. B. auf /monitor
    if (!rtspInput && !procInput && !confInput && !imgInput && !fwInput && !fhInput &&
        !couplingInput && !mqttEnabled && !mqttHost && !mqttPort && !mqttTopic && !mqttPayload &&
        !mqttUser && !mqttPass) {
        return;
    }

    try {
        const res = await fetch('/config');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const cfg = await res.json();

        if (rtspInput && typeof cfg.rtsp_url === 'string') {
            rtspInput.value = cfg.rtsp_url;
        }
        if (procInput && typeof cfg.process_interval === 'number') {
            procInput.value = cfg.process_interval;
        }
        if (confInput && typeof cfg.conf_thresh === 'number') {
            confInput.value = cfg.conf_thresh;
        }
        if (imgInput && typeof cfg.yolo_imgsz === 'number') {
            imgInput.value = cfg.yolo_imgsz;
        }
        if (fwInput && typeof cfg.frame_width === 'number') {
            fwInput.value = cfg.frame_width;
        }
        if (fhInput && typeof cfg.frame_height === 'number') {
            fhInput.value = cfg.frame_height;
        }
        if (couplingInput && typeof cfg.coupling_distance_px === 'number') {
            couplingInput.value = cfg.coupling_distance_px;
        }

        const mqtt = cfg.mqtt || {};
        if (mqttEnabled) {
            mqttEnabled.checked = !!mqtt.enabled;
        }
        if (mqttHost && typeof mqtt.host === 'string') {
            mqttHost.value = mqtt.host;
        }
        if (mqttPort && typeof mqtt.port === 'number') {
            mqttPort.value = mqtt.port;
        }
        if (mqttTopic && typeof mqtt.topic === 'string') {
            mqttTopic.value = mqtt.topic;
        }
        if (mqttPayload && typeof mqtt.payload_on === 'string') {
            mqttPayload.value = mqtt.payload_on;
        }
        if (mqttUser && typeof mqtt.username === 'string') {
            mqttUser.value = mqtt.username;
        }
        // Passwort wird im Backend nur als bool zurückgegeben → hier leer lassen
        if (mqttPass) {
            mqttPass.value = '';
        }

    } catch (e) {
        console.error('Config laden Fehler:', e);
        const cfgStatus = document.getElementById('configStatus');
        if (cfgStatus) cfgStatus.textContent = 'Fehler beim Laden der Einstellungen: ' + e;
    }
}

function initConfigControls() {
    const btnSave = document.getElementById('btnSaveConfig');
    if (!btnSave) return;

    const rtspInput     = document.getElementById('configRtspUrl');
    const procInput     = document.getElementById('configProcessInterval');
    const confInput     = document.getElementById('configConfThresh');
    const imgInput      = document.getElementById('configImgSz');
    const fwInput       = document.getElementById('configFrameWidth');
    const fhInput       = document.getElementById('configFrameHeight');
    const couplingInput = document.getElementById('configCouplingDistancePx');

    const mqttEnabled = document.getElementById('configMqttEnabled');
    const mqttHost    = document.getElementById('configMqttHost');
    const mqttPort    = document.getElementById('configMqttPort');
    const mqttTopic   = document.getElementById('configMqttTopic');
    const mqttPayload = document.getElementById('configMqttPayloadOn');
    const mqttUser    = document.getElementById('configMqttUser');
    const mqttPass    = document.getElementById('configMqttPass');

    const cfgStatus = document.getElementById('configStatus');

    btnSave.addEventListener('click', async () => {
        const payload = {};

        if (rtspInput && rtspInput.value.trim() !== '') {
            payload.rtsp_url = rtspInput.value.trim();
        }
        if (procInput && procInput.value !== '') {
            const v = parseFloat(procInput.value);
            if (!isNaN(v)) payload.process_interval = v;
        }
        if (confInput && confInput.value !== '') {
            const v = parseFloat(confInput.value);
            if (!isNaN(v)) payload.conf_thresh = v;
        }
        if (imgInput && imgInput.value !== '') {
            const v = parseInt(imgInput.value, 10);
            if (!isNaN(v)) payload.yolo_imgsz = v;
        }
        if (fwInput && fwInput.value !== '') {
            const v = parseInt(fwInput.value, 10);
            if (!isNaN(v)) payload.frame_width = v;
        }
        if (fhInput && fhInput.value !== '') {
            const v = parseInt(fhInput.value, 10);
            if (!isNaN(v)) payload.frame_height = v;
        }
        if (couplingInput && couplingInput.value !== '') {
            const v = parseFloat(couplingInput.value);
            if (!isNaN(v)) payload.coupling_distance_px = v;
        }

        // MQTT-Teil
        const mqttCfg = {};
        if (mqttEnabled) {
            mqttCfg.enabled = mqttEnabled.checked;
        }
        if (mqttHost && mqttHost.value.trim() !== '') {
            mqttCfg.host = mqttHost.value.trim();
        }
        if (mqttPort && mqttPort.value !== '') {
            const v = parseInt(mqttPort.value, 10);
            if (!isNaN(v)) mqttCfg.port = v;
        }
        if (mqttTopic && mqttTopic.value.trim() !== '') {
            mqttCfg.topic = mqttTopic.value.trim();
        }
        if (mqttPayload && mqttPayload.value.trim() !== '') {
            mqttCfg.payload_on = mqttPayload.value.trim();
        }
        if (mqttUser && mqttUser.value.trim() !== '') {
            mqttCfg.username = mqttUser.value.trim();
        }
        if (mqttPass && mqttPass.value !== '') {
            // nur setzen, wenn tatsächlich ein Wert eingegeben wurde
            mqttCfg.password = mqttPass.value;
        }

        if (Object.keys(mqttCfg).length > 0) {
            payload.mqtt = mqttCfg;
        }

        if (Object.keys(payload).length === 0) {
            if (cfgStatus) cfgStatus.textContent = 'Keine Werte geändert.';
            return;
        }

        try {
            const res = await fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok || data.ok === false) {
                throw new Error('HTTP ' + res.status);
            }
            if (cfgStatus) cfgStatus.textContent = 'Einstellungen übernommen.';
        } catch (e) {
            console.error('Config speichern Fehler:', e);
            if (cfgStatus) cfgStatus.textContent = 'Fehler beim Speichern: ' + e;
        }
    });

    // initial Werte laden
    loadConfig();
}

// ---------------- Init ----------------

document.addEventListener('DOMContentLoaded', () => {
    // Status-Polling für alle Views
    updateStatus();
    setInterval(updateStatus, 1000);

    // Config-Controls nur auf der Debug-Seite (/index)
    initConfigControls();
});
