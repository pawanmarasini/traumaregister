
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

// --- START Environment/Configuration (Simulated) ---
const GEMINI_API_KEY = process.env.API_KEY;
const PADDLE_OCR_ENDPOINT = 'http://localhost:8001/ocr/paddle'; // Replace with your actual PaddleOCR server endpoint
// --- END Environment/Configuration ---

// --- START Encryption Configuration ---
const APP_SECRET_SALT = 'KAHS-trauma-registry-super-secret-salt-v1.0.1'; // Store securely in a real app
const ENCRYPTED_DATA_KEY = 'traumaRegisterEncryptedDataV1';
let derivedEncryptionKey: CryptoKey | null = null;
// --- END Encryption Configuration ---


// --- START Globals and State ---
let currentUser: User | null = null;
let currentView: string = 'login'; // 'login', 'patient-details', 'treatment-investigations', 'admin-dashboard', 'ocr-error-log', 'view-records', 'nurse-upload'
let editingRecordId: string | null = null;
let ocrErrorToRetry: AdminErrorLogEntry | null = null;
let currentPatientForTreatment: TraumaRecord | null = null; // For treatment tab

// Nurse view specific state
let currentNurseView: 'patientDocUpload' | 'labReportUpload' | 'viewLabHistory' = 'patientDocUpload';
let currentActivePatientIdForNurse: string | null = null; // For nurse lab upload linking & viewing history
let nurseLabPatientSearchResults: TraumaRecord[] = []; // For nurse lab patient selection

const MAX_CLINICAL_IMAGES = 5;
const CLINICAL_IMAGE_PREVIEW_SIZE = 150; // px for width/height of card
const CLINICAL_IMAGE_MAX_DIMENSION = 1024; // px for compression
const CLINICAL_IMAGE_COMPRESSION_QUALITY = 0.75; // 0.0 to 1.0

const MAX_XRAY_IMAGES = 5;
const XRAY_IMAGE_PREVIEW_SIZE = 150; // px (can be different if needed)
const XRAY_IMAGE_MAX_DIMENSION = 1024; // px for compression
const XRAY_IMAGE_COMPRESSION_QUALITY = 0.75; // 0.0 to 1.0
const MAX_XRAY_FILE_SIZE_BYTES = 2 * 1024 * 1024; // 2MB

const LARGE_FILE_THRESHOLD_BYTES = 5 * 1024 * 1024; // 5MB
const LARGE_FILE_GENERAL_COMPRESSION_MAX_DIM = 1920; // For non-OCR general large files
const LARGE_FILE_GENERAL_COMPRESSION_QUALITY = 0.65;
const LARGE_FILE_OCR_COMPRESSION_MAX_DIM = 1280; // For OCR-bound large files (patient doc, lab reports)
const LARGE_FILE_OCR_COMPRESSION_QUALITY = 0.7;

let isOffline: boolean = !navigator.onLine;
let tesseractWorker: any | null = null;
let tesseractInitializing = false;

declare var Tesseract: any; // Fix for Tesseract namespace/name not found error

interface User {
    id: string;
    username: string;
    passwordHash: string;
    role: 'admin' | 'user' | 'nurse';
    isActive: boolean;
    createdAt: string;
}

interface Age {
    years: number;
    months?: number;
    days?: number;
}

interface ExtractedDemographicsFromOCR {
    patientName?: string;
    patientId?: string;
    beemaNumber?: string;
    age?: Age;
    ageYears?: number;
    ageMonths?: number;
    ageDays?: number;
    sex?: 'Male' | 'Female' | 'Other';
    address?: string;
    contactNumber?: string;
}

interface ExtractedLabReportData {
    reportDate?: string;
    labParameters?: { [key: string]: string };
    rawText?: string;
    error?: string;
}

interface LabReportEntry {
    id: string;
    fileName: string;
    fileUrl: string;
    fileType: string;
    reportDate?: string;
    labParameters?: { [key: string]: string };
    rawOcrText?: string;
    source: 'auto-captured' | 'manual-entry';
    createdAt: string;
}


interface TraumaRecord {
    id: string; // System generated record ID (e.g., PKAHS123-JohnDoe-30-Male-timestamp)
    patientId: string; // KAHS-prefixed Patient ID (e.g., KAHS12345, can be OCRed or generated)
    beemaNumber?: string;
    patientName: string;
    age: Age;
    sex: 'Male' | 'Female' | 'Other';
    address: string;
    contactNumber: string;
    patientDocumentUrl?: string;
    patientDocumentOcrText?: string;
    chiefComplaints?: string;
    dateOfInjuryAD: string;
    dateOfInjuryBS: string;
    timeOfInjury: string;
    modeOfInjury: string;
    otherMOI?: string;
    siteOfInjury: string;
    typeOfInjury: string;
    descriptionOfInjuries?: string;
    xrayImageUrls?: string[];
    presentComplaint: string;
    glasgowComaScale: string;
    vitalSigns: string;
    systemicExamination: string;
    localExamination: string;
    diagnosisSide?: ('Right' | 'Left')[];
    provisionalDiagnosis: string;
    clinicalImageUrls?: string[];
    labReportFiles?: LabReportEntry[];
    createdBy: string;
    createdAt: string;
    updatedAt?: string;
    updatedBy?: string; // To track who last updated the record
    conservativeTreatmentGiven?: string;
    operativeDateOfSurgery?: string;
    operativeNameOfSurgery?: string;
    operativeApproach?: string;
    operativeImplantUsed?: string;
    operativeNotes?: string;
    radiologicalInvestigationDetails?: string;
    manualLabNotes?: string;
    finalDiagnosisTreatment?: string;
    dischargeConditionOfWound?: string;
    dischargePulse?: string;
    dischargeBloodPressure?: string;
    dischargeTemperature?: string;
    dischargeRespiratoryRate?: string;
    dischargeAntibiotics?: MedicationEntry[];
    dischargeAnalgesics?: MedicationEntry[];
    dischargeOtherMedications?: MedicationEntry[];
    dischargeDietaryAdvice?: string;
    dischargeWoundCareAdvice?: string;
    dischargeDateSutureOut?: string;
    dischargeNextOpdVisit?: string;
    dischargeDressingAdvice?: string;
    physiotherapyRehabProtocol?: string;
    weightBearingAdvice?: string;
    exerciseProtocol?: string;
    restLimbElevationAdvice?: string;
}


interface MedicationEntry {
    id: string;
    drugName: string;
    dose: string;
    route: string;
    frequency: string;
    duration?: string;
}

interface AdminErrorLogEntry {
    id: string;
    timestamp: string;
    errorType: 'OCR_FAILURE' | 'OCR_API_KEY_MISSING' | 'OTHER' | 'PATIENT_DOC_OCR_FAILURE' | 'OCR_OFFLINE_TESSERACT_FAILURE' | 'LAB_REPORT_OCR_FAILURE' | 'ENCRYPTION_ERROR' | 'DECRYPTION_ERROR' | 'NURSE_UPLOAD_OCR_FAILURE' | 'GEMINI_OCR_FAILURE' | 'TESSERACT_OCR_FAILURE' | 'PADDLE_OCR_FAILURE' | 'GEMINI_CLIENT_INIT_FAILURE';
    message: string;
    fileName?: string;
    recordId?: string; // Internal record ID
    patientId?: string; // KAHS Patient ID, if available
    status: 'new' | 'resolved';
    originalFile?: File;
    uploadedBy?: string; // User ID of uploader (e.g., nurse)
}

// Simulated Database / Initial State
let initialUsers: User[] = [
    { id: 'admin001', username: 'pawanmarasini', passwordHash: hashPassword('admin123'), role: 'admin', isActive: true, createdAt: new Date().toISOString() },
    { id: 'user001', username: 'testuser', passwordHash: hashPassword('test123'), role: 'user', isActive: true, createdAt: new Date().toISOString() },
    { id: 'nurse001', username: 'testnurse', passwordHash: hashPassword('nurse123'), role: 'nurse', isActive: true, createdAt: new Date().toISOString() },
];
let users: User[] = [...initialUsers];
let traumaRecords: TraumaRecord[] = [];
let adminErrorLog: AdminErrorLogEntry[] = [];
// --- END Globals and State ---

// --- START Encryption/Decryption Helpers (Web Crypto API) ---
async function deriveKey(passwordStr: string, saltStr: string): Promise<CryptoKey> {
    const encoder = new TextEncoder();
    const passwordEncoded = encoder.encode(passwordStr);
    const saltEncoded = encoder.encode(saltStr);

    const masterKey = await crypto.subtle.importKey(
        "raw",
        passwordEncoded,
        "PBKDF2",
        false,
        ["deriveKey"]
    );

    return crypto.subtle.deriveKey(
        {
            name: "PBKDF2",
            salt: saltEncoded,
            iterations: 100000,
            hash: "SHA-256"
        },
        masterKey,
        { name: "AES-GCM", length: 256 },
        true,
        ["encrypt", "decrypt"]
    );
}

interface EncryptedPayload {
    iv: number[];
    content: number[];
}

async function encryptData(dataObject: any, key: CryptoKey): Promise<string | null> {
    try {
        const iv = crypto.getRandomValues(new Uint8Array(12));
        const dataString = JSON.stringify(dataObject);
        const encodedData = new TextEncoder().encode(dataString);

        const encryptedBuffer = await crypto.subtle.encrypt(
            { name: "AES-GCM", iv: iv },
            key,
            encodedData
        );

        const payload: EncryptedPayload = {
            iv: Array.from(iv),
            content: Array.from(new Uint8Array(encryptedBuffer))
        };
        return JSON.stringify(payload);
    } catch (error) {
        console.error("Encryption failed:", error);
        logAdminError('ENCRYPTION_ERROR', `Failed to encrypt data: ${error instanceof Error ? error.message : String(error)}`);
        showToast("Error: Could not encrypt data for saving.", "error");
        return null;
    }
}

async function decryptData(encryptedString: string, key: CryptoKey): Promise<any | null> {
    try {
        const payload: EncryptedPayload = JSON.parse(encryptedString);
        const iv = new Uint8Array(payload.iv);
        const encryptedContent = new Uint8Array(payload.content);

        const decryptedBuffer = await crypto.subtle.decrypt(
            { name: "AES-GCM", iv: iv },
            key,
            encryptedContent
        );

        const decryptedString = new TextDecoder().decode(decryptedBuffer);
        return JSON.parse(decryptedString);
    } catch (error) {
        console.error("Decryption failed:", error);
        logAdminError('DECRYPTION_ERROR', `Failed to decrypt data. Data might be corrupted or key is incorrect. ${error instanceof Error ? error.message : String(error)}`);
        showToast("Error: Could not decrypt stored data. It might be corrupted or the key is incorrect.", "error");
        return null;
    }
}

async function encryptAndSaveAllData(): Promise<void> {
    if (!derivedEncryptionKey) {
        showToast("Warning: Encryption key not available. Data cannot be saved securely.", "warning");
        console.warn("Attempted to save data, but no encryption key is available.");
        return;
    }

    const appState = {
        users,
        traumaRecords,
        adminErrorLog
    };

    const encryptedState = await encryptData(appState, derivedEncryptionKey);
    if (encryptedState) {
        try {
            localStorage.setItem(ENCRYPTED_DATA_KEY, encryptedState);
        } catch (e) {
            console.error("Failed to save encrypted data to localStorage:", e);
            showToast("Error: Failed to save data to local storage. Storage might be full.", "error");
            logAdminError('OTHER', `Failed to save to localStorage: ${e instanceof Error ? e.message : String(e)}`);
        }
    }
}

async function loadAndDecryptAllData(): Promise<void> {
    if (!derivedEncryptionKey) {
        console.warn("Attempted to load data, but no encryption key is available.");
        return;
    }

    const encryptedState = localStorage.getItem(ENCRYPTED_DATA_KEY);
    if (encryptedState) {
        const decryptedState = await decryptData(encryptedState, derivedEncryptionKey);
        if (decryptedState) {
            users = decryptedState.users || [...initialUsers];
            traumaRecords = decryptedState.traumaRecords || [];
            adminErrorLog = decryptedState.adminErrorLog || [];
            showToast("Application data loaded and decrypted.", "success");
        } else {
            showToast("Failed to load stored data. Using defaults.", "warning");
            users = [...initialUsers];
            traumaRecords = [];
            adminErrorLog = [];
        }
    } else {
        users = [...initialUsers];
        traumaRecords = [];
        adminErrorLog = [];
    }
}
// --- END Encryption/Decryption Helpers ---


// --- START AD/BS Date Converter (Simulation) ---
const adbs = {
    ad2bs: (adDateString: string): string => {
        if (!adDateString) return '';
        try {
            const date = new Date(adDateString);
            if (isNaN(date.getTime())) return 'Invalid AD Date';
            let bsYear = date.getFullYear() + 56;
            let bsMonth = date.getMonth() + 1 + 8;
            let bsDay = date.getDate() + 17;

            if (bsDay > 30) {
                bsDay -= 30;
                bsMonth += 1;
            }
            if (bsMonth > 12) {
                bsMonth -= 12;
                bsYear += 1;
            }
            const finalBsMonth = String(bsMonth).padStart(2, '0');
            const finalBsDay = String(bsDay).padStart(2, '0');

            return `${bsYear}/${finalBsMonth}/${finalBsDay}`;
        } catch (e) {
            return 'Conversion Error';
        }
    },
    bs2ad: (bsDateString: string): string => {
        if (!bsDateString) return '';
        try {
            const parts = bsDateString.split(/[\/-]/);
            if (parts.length !== 3) return 'Invalid BS Date';
            let adYear = parseInt(parts[0]) - 56;
            let adMonth = parseInt(parts[1]) - 8;
            let adDay = parseInt(parts[2]) - 17;

            if (adDay <= 0) { adDay += 30; adMonth -=1; }
            if (adMonth <= 0) { adMonth += 12; adYear -=1; }
            const date = new Date(adYear, adMonth -1, adDay);
             if (isNaN(date.getTime())) return 'Conversion Error';
            return date.toISOString().split('T')[0];
        } catch (e) { return 'Conversion Error'; }
    }
};

function parseBsDate(bsDateStr: string): { year: number; month: number; day: number } | null {
    if (!bsDateStr) return null;
    const parts = bsDateStr.split(/[\/-]/);
    if (parts.length === 3) {
        const year = parseInt(parts[0]);
        const month = parseInt(parts[1]);
        const day = parseInt(parts[2]);
        if (!isNaN(year) && !isNaN(month) && !isNaN(day) && month >= 1 && month <= 12 && day >=1 && day <= 32) {
            return { year, month, day };
        }
    }
    console.warn("Could not parse BS Date:", bsDateStr);
    return null;
}

function bsDateToAge(dobBsString: string, currentBsDateString?: string): Age | undefined {
    const dob = parseBsDate(dobBsString);
    if (!dob) return undefined;

    const todayAd = new Date().toISOString().split('T')[0];
    const currentBsDateToUse = currentBsDateString || adbs.ad2bs(todayAd);
    const currentBs = parseBsDate(currentBsDateToUse);

    if (!currentBs) {
      console.warn("Could not determine current BS date for age calculation.");
      return undefined;
    }

    let ageYears = currentBs.year - dob.year;
    let ageMonths = currentBs.month - dob.month;
    let ageDays = currentBs.day - dob.day;

    if (ageDays < 0) {
        ageMonths--;
        ageDays += 30;
    }
    if (ageMonths < 0) {
        ageYears--;
        ageMonths += 12;
    }

    if (ageYears < 0) {
        console.warn("Calculated negative age, returning undefined.", { dob, currentBs });
        return undefined;
    }

    return {
        years: ageYears,
        months: ageMonths >= 0 ? ageMonths : undefined,
        days: ageDays >= 0 ? ageDays : undefined,
    };
}

function formatAge(age?: Age): string {
    if (!age || age.years === undefined) return 'N/A'; // Ensure years is defined
    const parts = [];
    if (typeof age.years === 'number') parts.push(`${age.years}Y`);
    if (typeof age.months === 'number' && age.months > 0) parts.push(`${age.months}M`);
    if (typeof age.days === 'number' && age.days > 0) parts.push(`${age.days}D`);

    if (parts.length === 0 && typeof age.years === 'number') return `${age.years}Y`; // Case for 0M 0D
    return parts.length > 0 ? parts.join(' ') : (typeof age.years === 'number' ? `${age.years}Y` : 'N/A');
}

function getComparableDate(reportDateStr?: string): string {
    if (!reportDateStr) return '1970-01-01T00:00:00.000Z';
    if (reportDateStr.includes('(BS)')) {
        const bsDate = reportDateStr.replace(/\s*\(BS\)/i, '').trim();
        const adDate = adbs.bs2ad(bsDate);
        return adDate !== 'Conversion Error' && adDate !== 'Invalid BS Date' ? new Date(adDate).toISOString() : '1970-01-01T00:00:00.000Z';
    }
    const parsedAD = new Date(reportDateStr);
    if (!isNaN(parsedAD.getTime())) {
        return parsedAD.toISOString();
    }
    return '1970-01-01T00:00:00.000Z';
}
// --- END AD/BS Date Converter ---

// --- START Tesseract.js OCR ---
async function initializeTesseractWorker() {
    if (tesseractWorker || tesseractInitializing) return;
    tesseractInitializing = true;
    try {
        const worker = await Tesseract.createWorker('eng', 1, { // OEM_LSTM_ONLY
            logger: m => { if(m.status === 'recognizing text') console.log(`Tesseract: ${m.status} (${(m.progress * 100).toFixed(0)}%)`); },
            workerPath: 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.0/dist/worker.min.js',
            langPath: 'https://tessdata.projectnaptha.com/4.0.0_best', // Using best quality LSTM data
            corePath: 'https://cdn.jsdelivr.net/npm/tesseract.js-core@5.1.1/', // Base path for core files
        });
        tesseractWorker = worker;
        console.log("Tesseract.js worker initialized successfully with explicit paths.");
        showToast("Local OCR engine ready.", "info");
    } catch (error) {
        console.error("Failed to initialize Tesseract.js worker:", error);
        let detailedMessage = `Tesseract.js worker initialization failed: ${error instanceof Error ? error.message : String(error)}`;
        let toastMessage = "Failed to initialize local OCR engine. OCR may be limited. Check console.";

        if (error instanceof Error && (error.message.includes("importScripts") || error.message.includes("cdn.jsdelivr.net") || error.message.includes("failed to load"))) {
            detailedMessage += "\nThis might be due to network issues preventing access to CDN resources (e.g., tesseract.js-core from cdn.jsdelivr.net), or Content Security Policy restrictions. Please check your internet connection, browser console for more details. If using an ad-blocker, try disabling it for this site.";
            toastMessage = "Local OCR (Tesseract) failed to load resources from CDN. Check network/CSP. Fallbacks active.";
            console.warn("Tesseract.js CDN Load Failure Details: The browser's worker environment could not load necessary scripts from the CDN. This could be due to: 1. Network connectivity issues (check internet, firewall). 2. Content Security Policy (CSP) on the server blocking scripts from this CDN. 3. Browser extensions (like ad-blockers) interfering. 4. Temporary CDN outage. Ensure cdn.jsdelivr.net and tessdata.projectnaptha.com are accessible.");
        }
        logAdminError('TESSERACT_OCR_FAILURE', detailedMessage); // Changed error type
        showToast(toastMessage, "error");
        tesseractWorker = null;
    } finally {
        tesseractInitializing = false;
    }
}

async function performOcrWithTesseract(file: File, context: 'patient_document'): Promise<{ text?: string; error?: string, extractedDemographics?: Partial<ExtractedDemographicsFromOCR> }> {
    if (context !== 'patient_document') {
        return { error: "Tesseract OCR is currently only configured for patient documents." };
    }

    if (!tesseractWorker && !tesseractInitializing) {
        await initializeTesseractWorker();
    }
    if (!tesseractWorker) {
        return { error: "Tesseract OCR worker not available. Initialization might have failed (check console)." };
    }

    try {
        const { data: { text } } = await tesseractWorker.recognize(file);
        const demographics: Partial<ExtractedDemographicsFromOCR> = {};
        const lines = text.split('\n');

        for (const line of lines) {
            const l = line.toLowerCase();

            if (!demographics.patientId) {
                const idMatch = line.match(/(KAHS\d+)/i);
                if (idMatch && idMatch[1]) demographics.patientId = idMatch[1].toUpperCase();
            }

            if (!demographics.beemaNumber) {
                const beemaKeywords = /(?:NHIS| बीमा|Beema|Insurance)(?:\s*(?:No\.?|नं\.?))?\s*[:\-—]?\s*(\d{9,12})/i;
                const beemaKeywordMatch = line.match(beemaKeywords);
                if (beemaKeywordMatch && beemaKeywordMatch[1]) {
                    demographics.beemaNumber = beemaKeywordMatch[1];
                } else {
                    const standaloneBeemaMatch = line.match(/\b(\d{3}-?\d{3}-?\d{3}|\d{9,12})\b/);
                    if (standaloneBeemaMatch && standaloneBeemaMatch[1] && standaloneBeemaMatch[1].replace(/-/g, '').length >= 9) {
                        if (!line.match(/\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}/) && !l.includes('contact') && !l.includes('phone') && !l.includes('id')) {
                           demographics.beemaNumber = standaloneBeemaMatch[1].replace(/-/g, '');
                        }
                    }
                }
            }

            if (demographics.ageYears === undefined) {
                const dobMatchBS = line.match(/(?:जन्म\s*मिति|DOB.?\s*\(?BS\)?)\s*[:\-—]?\s*(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})/i);
                if (dobMatchBS && dobMatchBS[1]) {
                    const calculatedAge = bsDateToAge(dobMatchBS[1]);
                    if (calculatedAge && typeof calculatedAge.years === 'number') {
                        demographics.ageYears = calculatedAge.years;
                        if (calculatedAge.months !== undefined) demographics.ageMonths = calculatedAge.months;
                        if (calculatedAge.days !== undefined) demographics.ageDays = calculatedAge.days;
                    }
                } else {
                    const ageYMDMatch = line.match(/(?:Age|उमेर)\s*[:\-—]?\s*(?:(\d+)\s*Y(?:ears?)?)?\s*(?:(\d+)\s*M(?:onths?)?)?\s*(?:(\d+)\s*D(?:ays?)?)?/i);
                    if (ageYMDMatch) {
                        const years = ageYMDMatch[1] ? parseInt(ageYMDMatch[1]) : undefined;
                        const months = ageYMDMatch[2] ? parseInt(ageYMDMatch[2]) : undefined;
                        const days = ageYMDMatch[3] ? parseInt(ageYMDMatch[3]) : undefined;
                        if (years !== undefined) {
                            demographics.ageYears = years;
                            if (months !== undefined) demographics.ageMonths = months;
                            if (days !== undefined) demographics.ageDays = days;
                        }
                    }
                }
            }

            if (!demographics.sex) {
                const sexMatch = line.match(/\b(Male|Female|Other|पुरुष|महिला|अन्य)\b/i);
                if (sexMatch && sexMatch[1]) {
                    const s = sexMatch[1].toLowerCase();
                    if (s === 'male' || s === 'पुरुष') demographics.sex = 'Male';
                    else if (s === 'female' || s === 'महिला') demographics.sex = 'Female';
                    else if (s === 'other' || s === 'अन्य') demographics.sex = 'Other';
                }
            }

            if (!demographics.address) {
                const addressKeywordMatch = line.match(/^(?:Address|ठेगाना|स्थान)\s*[:\-—]?\s*(.+)/i);
                if (addressKeywordMatch && addressKeywordMatch[1]) {
                     demographics.address = addressKeywordMatch[1].trim();
                } else {
                    const commonLocationTerms = /\b(?:RM|Gaunpalika|Nagarpalika|Ward|टोल|गाउँ|जिल्ला|District|Province|प्रदेश)\b/i;
                    const nonAddressKeywords = /\b(?:Phone|Contact|ID|Number|No\.|Name|Age|Sex|DOB|Date)\b/i;
                    if (commonLocationTerms.test(line) && !nonAddressKeywords.test(l) && line.length > 10) {
                        const previousLine = lines[lines.indexOf(line) -1];
                        if(previousLine && /^(?:Address|ठेगाना|स्थान)\s*[:\-—]?\s*$/i.test(previousLine.trim())) {
                            demographics.address = (demographics.address ? demographics.address + ", " : "") + line.trim();
                        } else if (!demographics.address) {
                             demographics.address = line.trim();
                        }
                    }
                }
            }


            if (!demographics.contactNumber) {
                 const contactKeywordMatch = line.match(/(?:Contact|Phone|फोन|मोबाइल)(?:\s*(?:No\.?|नं\.?))?\s*[:\-—]?\s*((?:\+977[- ]?)?9\d{9})\b/i);
                 if (contactKeywordMatch && contactKeywordMatch[1]) {
                     let num = contactKeywordMatch[1].replace(/[- ]/g, '');
                     if (!num.startsWith('+977')) num = `+977${num}`;
                     demographics.contactNumber = num;
                 } else {
                     const standaloneContactMatch = line.match(/\b((?:\+977[- ]?)?9\d{9})\b/);
                     if (standaloneContactMatch && standaloneContactMatch[1]) {
                         if (!l.includes('id') && !l.includes('beema') && !l.includes('nhis')) {
                            let num = standaloneContactMatch[1].replace(/[- ]/g, '');
                            if (!num.startsWith('+977')) num = `+977${num}`;
                            demographics.contactNumber = num;
                         }
                     }
                 }
            }
        }

        if (!demographics.patientId && demographics.ageYears === undefined) {
             logAdminError('TESSERACT_OCR_FAILURE', 'Tesseract OCR completed but extracted minimal useful data (no Patient ID or Age).', file.name, undefined, undefined, file, currentUser?.id);
        }

        return { text, extractedDemographics: demographics };
    } catch (error: any) {
        console.error("Tesseract.js OCR error:", error);
        let errorMessage = "Tesseract OCR failed.";
        if (error.message) errorMessage += ` Details: ${error.message}`;
        logAdminError('TESSERACT_OCR_FAILURE', errorMessage, file.name, undefined, undefined, file, currentUser?.id);
        return { error: errorMessage, text: `Tesseract Error: ${errorMessage}` };
    }
}
// --- END Tesseract.js OCR ---

// --- START PaddleOCR ---
async function performOcrWithPaddle(file: File, context: 'patient_document' | 'lab_report'): Promise<{
    rawText?: string;
    error?: string;
    extractedDemographics?: Partial<ExtractedDemographicsFromOCR>;
    extractedLabData?: Partial<ExtractedLabReportData>;
}> {
    showToast("Attempting OCR with PaddleOCR (fallback)...", "info");
    const formData = new FormData();
    formData.append('image', file);
    formData.append('context', context); // Send context to help PaddleOCR server if needed

    try {
        const response = await fetch(PADDLE_OCR_ENDPOINT, {
            method: 'POST',
            body: formData,
            // Add any necessary headers, e.g., for API keys if your PaddleOCR server requires them
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`PaddleOCR server error: ${response.status} ${response.statusText}. Details: ${errorText.substring(0, 200)}`);
        }

        const resultJson = await response.json();
        const extractedDemographics: Partial<ExtractedDemographicsFromOCR> = {};
        const extractedLabData: Partial<ExtractedLabReportData> = {};

        if (resultJson.name && resultJson.name.toLowerCase() !== 'not found') extractedDemographics.patientName = resultJson.name;
        if (resultJson.patient_id && resultJson.patient_id.toLowerCase() !== 'not found') extractedDemographics.patientId = resultJson.patient_id;
        if (resultJson.age && typeof resultJson.age.years === 'number') {
            extractedDemographics.age = {
                years: resultJson.age.years,
                months: resultJson.age.months,
                days: resultJson.age.days
            };
        }
        if (resultJson.sex && resultJson.sex.toLowerCase() !== 'not found') extractedDemographics.sex = resultJson.sex as 'Male' | 'Female' | 'Other';
        if (resultJson.address && resultJson.address.toLowerCase() !== 'not found') extractedDemographics.address = resultJson.address;
        if (resultJson.contact && resultJson.contact.toLowerCase() !== 'not found') {
            let contact = String(resultJson.contact).replace(/\s+/g, '');
            if (contact.length === 10 && !contact.startsWith('+977')) {
                extractedDemographics.contactNumber = `+977${contact}`;
            } else if (contact.startsWith('+977') && contact.length >= 13) {
                 extractedDemographics.contactNumber = contact;
            } else {
                 extractedDemographics.contactNumber = contact; // Store as is if format is unusual
            }
        }
        if (resultJson.nhis_beema_no && String(resultJson.nhis_beema_no).toLowerCase() !== 'not found') extractedDemographics.beemaNumber = String(resultJson.nhis_beema_no);

        if (context === 'lab_report' && resultJson.lab_data) {
            if (resultJson.lab_data.report_date && String(resultJson.lab_data.report_date).toLowerCase() !== 'not found') {
                extractedLabData.reportDate = String(resultJson.lab_data.report_date);
            }
            if (resultJson.lab_data.parameters && Object.keys(resultJson.lab_data.parameters).length > 0) {
                extractedLabData.labParameters = resultJson.lab_data.parameters;
            }
        }

        const rawText = resultJson.raw_text || undefined;
        if (extractedLabData.labParameters || extractedLabData.reportDate) extractedLabData.rawText = rawText;


        if (Object.keys(extractedDemographics).length === 0 && Object.keys(extractedLabData).length === 0 && !rawText) {
             return { error: "PaddleOCR extracted no data.", rawText };
        }

        return {
            rawText,
            extractedDemographics: Object.keys(extractedDemographics).length > 0 ? extractedDemographics : undefined,
            extractedLabData: Object.keys(extractedLabData).length > 0 ? extractedLabData : undefined,
        };

    } catch (error: any) {
        console.error("PaddleOCR API error:", error);
        return { error: `PaddleOCR request failed: ${error.message}` };
    }
}
// --- END PaddleOCR ---


// --- START Gemini API Client ---
let ai: GoogleGenAI | null = null;
if (GEMINI_API_KEY) {
    try {
        ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
    } catch (error) {
        console.error("Failed to initialize GoogleGenAI:", error);
        ai = null;
        logAdminError('GEMINI_CLIENT_INIT_FAILURE', 'Failed to initialize Gemini API client.');
    }
} else {
    console.warn("GEMINI_API_KEY is not set. OCR functionality will be limited/simulated.");
}

// Helper to create mock OCR data
function getMockOcrData(context: 'lab_report' | 'patient_document', errorPrefix: string = "Mock OCR"): {
    rawText?: string;
    error?: string;
    extractedDemographics?: Partial<ExtractedDemographicsFromOCR>;
    extractedLabData?: Partial<ExtractedLabReportData>;
} {
    if (context === 'patient_document') {
        return {
            rawText: `${errorPrefix}: FullName: John Doe, PatientID: KAHS123MOCK, Age: Y 30 M 0 D 0, Sex: Male, BeemaNo: 123-456-789, Address: Mock Address, ContactNo: 9800000000`,
            extractedDemographics: { patientName: "John Doe (Mock)", patientId: "KAHS123MOCK", age: { years: 30, months: 0, days: 0 }, sex: "Male", beemaNumber: "123-456-789", address: "Mock Address", contactNumber: "+9779800000000" },
            error: errorPrefix.includes("Mock") ? undefined : errorPrefix
        };
    } else { // lab_report
        return {
            rawText: `${errorPrefix}: ReportDateBS: 2080-01-01, LabParametersJSON: {\"Hemoglobin\": \"10 g/dL\", \"WBC\": \"5000 /cumm\"}`,
            extractedLabData: {
                reportDate: "2080-01-01 (BS)",
                labParameters: { "Hemoglobin": "10 g/dL (Mock)", "WBC": "5000 /cumm (Mock)" }
            },
            error: errorPrefix.includes("Mock") ? undefined : errorPrefix
        };
    }
}


async function performOcr(file: File, context: 'lab_report' | 'patient_document'): Promise<{
    rawText?: string;
    error?: string;
    extractedDemographics?: Partial<ExtractedDemographicsFromOCR>;
    extractedLabData?: Partial<ExtractedLabReportData>;
}> {
    let ocrProcessingIndicator: HTMLDivElement | null = null;
    if (currentUser?.role === 'nurse') {
        // Updated selector for nurse view based on active section
        if (currentNurseView === 'patientDocUpload') {
            ocrProcessingIndicator = S<HTMLDivElement>('#nurse-patient-doc-ocr-indicator');
        } else if (currentNurseView === 'labReportUpload') {
            ocrProcessingIndicator = S<HTMLDivElement>('#nurse-lab-ocr-indicator');
        }
    } else {
        ocrProcessingIndicator = S<HTMLDivElement>(context === 'patient_document' ? '#patient-doc-ocr-indicator' : '#lab-ocr-indicator');
    }
    if (ocrProcessingIndicator) ocrProcessingIndicator.style.display = 'flex';

    const uploadedBy = currentUser?.id;
    const commonLogParams = { fileName: file.name, recordId: undefined, patientId: undefined, originalFile: file, uploadedBy };

    // 1. Try Gemini
    if (ai && !isOffline) { // Gemini only if online and initialized
        try {
            showToast("Attempting OCR with Gemini AI...", "info");
            const base64Data = await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });

            const imagePart = { inlineData: { mimeType: file.type, data: base64Data } };
            let promptText = "";
            if (context === 'patient_document') {
                 promptText = `Extract patient demographic information from this document. Structure the output strictly as follows, ensuring each field is on a new line:
FullName: [Patient's Full Name]
PatientID: [Patient ID, must start with KAHS followed by numbers, e.g., KAHS12345. If not found, state 'Not Found'.]
Age: [Extract as 'Y <years> M <months> D <days>' format (e.g., 'Y 45 M 11 D 7') OR if a Bikram Sambat birth date (like 'जन्म मिति') is found, output as 'DOB_BS: YYYY-MM-DD'. If not found, state 'Not Found'.]
Sex: [Male, Female, or Other. If not found, state 'Not Found'.]
Address: [Full Nepali address including Province (प्रदेश), District (जिल्ला), Municipality/VDC, Ward/Tole/Sadak. If not found, state 'Not Found'.]
ContactNo: [10-digit contact number if found, without country code. If not found, state 'Not Found'.]
BeemaNo: [Beema/NHIS number in NNN-NNN-NNN format or as raw 9-12 digits. Look for keywords 'बीमा नं.', 'Beema Number', 'Insurance No', or hyperlinked 9-digit numbers, especially near terms like 'Payment', 'Claim Code', 'Insurance'. Typical location is top-right on cards. If not found, state 'Not Found'.]

If any field is not found, write 'Not Found' for that field. Ensure age components (years, months, days) are numeric.`;
            } else { // lab_report
                 promptText = `Extract the following information from this lab report:
1. Report Date: Find the date of the report. If it's a Bikram Sambat (BS) date (e.g., YYYY/MM/DD or YYYY-MM-DD, often near Nepali text or 'मिति'), output it as "ReportDateBS: YYYY-MM-DD". If it's an AD date, output as "ReportDateAD: YYYY-MM-DD".
2. Lab Parameters: Identify all lab test names and their corresponding values with units. Output them as a JSON string under the key "LabParametersJSON:". For example:
LabParametersJSON: {"Hemoglobin": "12.8 g/dL", "WBC": "7500 /mm³", "Platelets": "250x10^3/µL", "Serum Creatinine": "1.1 mg/dL", "CRP": "7 mg/L"}

If a field is not found, state "Not Found" for that field.
Example Output Structure:
ReportDateBS: 2080-05-15
LabParametersJSON: {"Hemoglobin": "13.0 g/dL", "Total Leukocyte Count": "8000 /cumm"}
--- End Example ---
Strictly follow this output format. Provide only the requested fields.`;
            }
            const textPart = { text: promptText };
            const response: GenerateContentResponse = await ai.models.generateContent({
                model: 'gemini-2.5-flash-preview-04-17',
                contents: { parts: [imagePart, textPart] },
            });
            const ocrTextFromGemini = response.text;

            if (ocrTextFromGemini) {
                let extractedDemographics: Partial<ExtractedDemographicsFromOCR> = {};
                let extractedLabData: Partial<ExtractedLabReportData> = {};
                 if (context === 'patient_document') {
                    const nameMatch = ocrTextFromGemini.match(/^FullName:\s*(.*)/im);
                    if (nameMatch && nameMatch[1].toLowerCase() !== 'not found') extractedDemographics.patientName = nameMatch[1].trim();
                    const patientIdMatch = ocrTextFromGemini.match(/^PatientID:\s*(KAHS\d+)/im);
                    if (patientIdMatch) extractedDemographics.patientId = patientIdMatch[1].trim();
                    const ageTextMatch = ocrTextFromGemini.match(/^Age:\s*(.*)/im);
                    if (ageTextMatch && ageTextMatch[1] && ageTextMatch[1].trim().toLowerCase() !== 'not found') {
                        const ageStr = ageTextMatch[1].trim();
                        if (ageStr.startsWith('DOB_BS:')) {
                            const dobBs = ageStr.replace('DOB_BS:', '').trim();
                            const calculatedAge = bsDateToAge(dobBs);
                            if (calculatedAge && typeof calculatedAge.years === 'number') extractedDemographics.age = calculatedAge;
                        } else {
                            const yearsMatch = ageStr.match(/Y\s*(\d+)/i); const monthsMatch = ageStr.match(/M\s*(\d+)/i); const daysMatch = ageStr.match(/D\s*(\d+)/i);
                            const ageObj: Partial<Age> = {};
                            if (yearsMatch) ageObj.years = parseInt(yearsMatch[1]); else ageObj.years = 0;
                            if (monthsMatch) ageObj.months = parseInt(monthsMatch[1]); if (daysMatch) ageObj.days = parseInt(daysMatch[1]);
                            if (typeof ageObj.years === 'number') extractedDemographics.age = ageObj as Age;
                        }
                    }
                    const sexMatch = ocrTextFromGemini.match(/^Sex:\s*(Male|Female|Other)/im);
                    if (sexMatch) extractedDemographics.sex = sexMatch[1].trim() as 'Male' | 'Female' | 'Other';
                    const addressMatch = ocrTextFromGemini.match(/^Address:\s*(.*)/im);
                    if (addressMatch && addressMatch[1].toLowerCase() !== 'not found') extractedDemographics.address = addressMatch[1].trim();
                    const contactNoMatch = ocrTextFromGemini.match(/^ContactNo:\s*(\d{10})/im);
                    if (contactNoMatch) extractedDemographics.contactNumber = `+977${contactNoMatch[1].trim()}`;
                    let beemaNoMatch = ocrTextFromGemini.match(/^BeemaNo:\s*(\d{3}-\d{3}-\d{3})/im);
                    if (beemaNoMatch && beemaNoMatch[1].toLowerCase() !== 'not found') extractedDemographics.beemaNumber = beemaNoMatch[1].trim();
                    else { beemaNoMatch = ocrTextFromGemini.match(/^BeemaNo:\s*(\d{9,12})/im); if (beemaNoMatch && beemaNoMatch[1].toLowerCase() !== 'not found') extractedDemographics.beemaNumber = beemaNoMatch[1].trim(); }
                } else { // lab_report
                    let reportDateMatch = ocrTextFromGemini.match(/^ReportDateBS:\s*(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})/im);
                    if (reportDateMatch && reportDateMatch[1] && reportDateMatch[1].toLowerCase() !== 'not found') extractedLabData.reportDate = `${reportDateMatch[1].trim()} (BS)`;
                    else { reportDateMatch = ocrTextFromGemini.match(/^ReportDateAD:\s*(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})/im); if (reportDateMatch && reportDateMatch[1] && reportDateMatch[1].toLowerCase() !== 'not found') extractedLabData.reportDate = reportDateMatch[1].trim(); }
                    const labParamsJsonMatch = ocrTextFromGemini.match(/LabParametersJSON:\s*({.*?})/ims);
                    if (labParamsJsonMatch && labParamsJsonMatch[1] && labParamsJsonMatch[1].toLowerCase() !== 'not found') {
                        try { let jsonStr = labParamsJsonMatch[1].trim(); const fenceRegex = /^```json\s*\n?(.*?)\n?\s*```$/s; const fenceContentMatch = jsonStr.match(fenceRegex); if (fenceContentMatch && fenceContentMatch[1]) jsonStr = fenceContentMatch[1].trim(); extractedLabData.labParameters = JSON.parse(jsonStr); }
                        catch (e) { console.error("Failed to parse LabParametersJSON from Gemini:", e); logAdminError(currentUser?.role === 'nurse' ? 'NURSE_UPLOAD_OCR_FAILURE' : 'LAB_REPORT_OCR_FAILURE', `Gemini: Failed to parse lab params. Error: ${e instanceof Error ? e.message : String(e)}`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy); extractedLabData.error = "Failed to parse lab parameters."; }
                    }
                    if (!extractedLabData.reportDate && !extractedLabData.labParameters) { logAdminError(currentUser?.role === 'nurse' ? 'NURSE_UPLOAD_OCR_FAILURE' : 'LAB_REPORT_OCR_FAILURE', `Gemini OCR for lab report extracted no structured data. Raw: ${ocrTextFromGemini.substring(0,100)}...`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy); extractedLabData.error = extractedLabData.error || "OCR extracted no structured lab data.";}
                }
                if (ocrProcessingIndicator) ocrProcessingIndicator.style.display = 'none';
                return { rawText: ocrTextFromGemini, extractedDemographics, extractedLabData, error: extractedLabData.error };
            }
        } catch (geminiError: any) {
            logAdminError('GEMINI_OCR_FAILURE', `Gemini API Error: ${geminiError.message}`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
            showToast("Gemini OCR failed. Trying fallbacks...", "warning");
        }
    } else if (isOffline) {
        showToast("Offline. Gemini AI not available.", "info");
    } else { // !ai (Gemini client not initialized)
        logAdminError('GEMINI_CLIENT_INIT_FAILURE', "Gemini AI client not initialized. API key might be missing.", commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
        showToast("Gemini AI not configured. Trying fallbacks...", "warning");
    }

    // 2. Try Tesseract (if applicable for context and available)
    if (context === 'patient_document') { // Tesseract only for patient documents in current setup
        if (tesseractWorker || tesseractInitializing) {
            try {
                showToast("Attempting OCR with Tesseract (fallback)...", "info");
                const tesseractResult = await performOcrWithTesseract(file, 'patient_document');
                if (tesseractResult && !tesseractResult.error && (tesseractResult.text || (tesseractResult.extractedDemographics && Object.keys(tesseractResult.extractedDemographics).length > 0))) {
                    if (ocrProcessingIndicator) ocrProcessingIndicator.style.display = 'none';
                    showToast("OCR via Tesseract successful.", "success");
                    return tesseractResult;
                }
                if (tesseractResult.error) {
                    logAdminError('TESSERACT_OCR_FAILURE', `Tesseract fallback failed: ${tesseractResult.error}`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
                }
            } catch (tesseractError: any) {
                logAdminError('TESSERACT_OCR_FAILURE', `Error during Tesseract fallback: ${tesseractError.message}`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
            }
        } else if (isOffline) {
            showToast("Offline, but local OCR (Tesseract) not ready.", "warning");
             logAdminError('TESSERACT_OCR_FAILURE', `Tesseract not initialized during offline attempt.`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
        }
    }

    // 3. Try PaddleOCR (if online and previous steps failed or weren't applicable)
    if (!isOffline) {
        try {
            const paddleResult = await performOcrWithPaddle(file, context);
            if (paddleResult && !paddleResult.error && (paddleResult.rawText || (paddleResult.extractedDemographics && Object.keys(paddleResult.extractedDemographics).length > 0) || (paddleResult.extractedLabData && Object.keys(paddleResult.extractedLabData).length > 0) )) {
                if (ocrProcessingIndicator) ocrProcessingIndicator.style.display = 'none';
                showToast("OCR via PaddleOCR successful.", "success");
                return paddleResult;
            }
            if(paddleResult.error) {
                logAdminError('PADDLE_OCR_FAILURE', `PaddleOCR fallback failed: ${paddleResult.error}`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
            }
        } catch (paddleError: any) {
            logAdminError('PADDLE_OCR_FAILURE', `Error during PaddleOCR fallback: ${paddleError.message}`, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
        }
    } else if (context !== 'patient_document' || !tesseractWorker) { // For lab reports offline, or if Tesseract failed for patient_document
         showToast("Offline: Cannot use PaddleOCR. Limited OCR options.", "warning");
    }


    // 4. All OCR attempts failed or not applicable, use mock data / final error
    if (ocrProcessingIndicator) ocrProcessingIndicator.style.display = 'none';
    const finalErrorMsg = "All OCR methods failed or are unavailable.";
    showToast(finalErrorMsg, "error");
    logAdminError(currentUser?.role === 'nurse' ? 'NURSE_UPLOAD_OCR_FAILURE' : (context === 'patient_document' ? 'PATIENT_DOC_OCR_FAILURE' : 'LAB_REPORT_OCR_FAILURE'), finalErrorMsg, commonLogParams.fileName, commonLogParams.recordId, commonLogParams.patientId, commonLogParams.originalFile, commonLogParams.uploadedBy);
    
    await new Promise(resolve => setTimeout(resolve, 500)); // Brief pause before showing mock
    return getMockOcrData(context, finalErrorMsg);
}
// --- END Gemini API Client ---


// --- START Utility Functions ---
function S<T extends HTMLElement>(selector: string): T | null { return document.querySelector<T>(selector); }
function SAll<T extends HTMLElement>(selector: string): NodeListOf<T> { return document.querySelectorAll<T>(selector); }
function uuid(): string { return crypto.randomUUID(); }

function hashPassword(password: string): string { return `hashed_${password}`; }
function verifyPassword(password: string, hash: string): boolean { return hashPassword(password) === hash; }

function showToast(message: string, type: 'success' | 'error' | 'info' | 'warning' = 'info') {
    const toast = S<HTMLDivElement>('#toast-notification');
    const toastMessage = S<HTMLDivElement>('#toast-message');
    if (toast && toastMessage) {
        toastMessage.textContent = message;
        toast.className = 'toast show';
        if (type === 'success') toast.style.backgroundColor = 'var(--success-color)';
        else if (type === 'error') toast.style.backgroundColor = 'var(--danger-color)';
        else if (type === 'warning') toast.style.backgroundColor = 'var(--warning-color, #ffc107)';
        else toast.style.backgroundColor = 'var(--dark-color)';
        setTimeout(() => { toast.className = 'toast'; }, type === 'error' || type === 'warning' ? 5000 : 3000); // Longer display for errors/warnings
    }
}

function openModal(title: string, contentHtml: string, size: 'default' | 'lg' = 'default', footerButtons: { confirmText?: string, cancelText?: string } | false = { confirmText: 'Confirm', cancelText: 'Cancel' }) {
    const modalContainer = S<HTMLDivElement>('#modal-container');
    const modalContent = modalContainer?.querySelector<HTMLDivElement>('.modal-content');
    const modalTitle = S<HTMLHeadingElement>('#modal-title');
    const modalBody = S<HTMLDivElement>('#modal-body');
    const modalFooter = S<HTMLDivElement>('#modal-footer');
    const modalConfirmBtn = S<HTMLButtonElement>('#modal-confirm-button');
    const modalCancelBtn = S<HTMLButtonElement>('#modal-cancel-button');

    if (modalContainer && modalContent && modalTitle && modalBody && modalFooter && modalConfirmBtn && modalCancelBtn) {
        modalTitle.textContent = title;
        modalBody.innerHTML = contentHtml;

        if (size === 'lg') {
            modalContent.classList.add('modal-lg');
        } else {
            modalContent.classList.remove('modal-lg');
        }

        if (footerButtons) {
            modalConfirmBtn.textContent = footerButtons.confirmText || 'Confirm';
            modalCancelBtn.textContent = footerButtons.cancelText || 'Cancel';
            modalConfirmBtn.style.display = 'inline-block';
            modalCancelBtn.style.display = 'inline-block';
            modalFooter.style.display = 'flex'; // Use flex for footer layout
        } else {
            modalConfirmBtn.style.display = 'none';
            modalCancelBtn.style.display = 'none';
            modalFooter.style.display = 'none';
        }
        modalContainer.style.display = 'flex';
        return { modalContainer, modalConfirmBtn, modalCancelBtn };
    }
    return null;
}

function closeModal() {
    const modalContainer = S<HTMLDivElement>('#modal-container');
    if (modalContainer) modalContainer.style.display = 'none';
}

async function compressImage(
    file: File,
    maxWidth: number,
    maxHeight: number,
    quality: number,
): Promise<string> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = () => {
            let { width, height } = img;
            if (width > height) {
                if (width > maxWidth) {
                    height = Math.round((height * maxWidth) / width);
                    width = maxWidth;
                }
            } else {
                if (height > maxHeight) {
                    width = Math.round((width * maxHeight) / height);
                    height = maxHeight;
                }
            }

            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                return reject(new Error('Failed to get canvas context'));
            }
            ctx.drawImage(img, 0, 0, width, height);

            let mimeType = 'image/jpeg';
            if (file.type === 'image/png' && quality >= 0.9) {
                mimeType = 'image/png';
            } else if (file.type === 'application/pdf') {
                console.warn("PDF compression to image requested, this will effectively convert first page to image. For full PDF compression, use a PDF library.");
                 mimeType = 'image/jpeg';
            }


            resolve(canvas.toDataURL(mimeType, quality));
            URL.revokeObjectURL(img.src);
        };
        img.onerror = (error) => {
            URL.revokeObjectURL(img.src);
            if (file.type === 'application/pdf') {
                console.warn("Direct canvas drawing of PDF failed as expected. Returning original PDF as base64 for storage/linking, no image compression applied.");
                const reader = new FileReader();
                reader.onloadend = () => {
                    resolve(reader.result as string);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
                return;
            }
            reject(error);
        };
    });
}

async function fileToDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}


async function base64StringToFile(base64String: string, fileName: string, mimeType?: string): Promise<File> {
    const res = await fetch(base64String);
    const blob = await res.blob();
    return new File([blob], fileName, { type: mimeType || blob.type });
}


function logAdminError(type: AdminErrorLogEntry['errorType'], message: string, fileName?: string, recordId?: string, patientId?: string, originalFile?: File, uploadedBy?: string) {
    const errorEntry: AdminErrorLogEntry = {
        id: uuid(),
        timestamp: new Date().toISOString(),
        errorType: type,
        message,
        fileName,
        recordId,
        patientId,
        status: 'new',
        originalFile,
        uploadedBy
    };
    adminErrorLog.push(errorEntry);
    console.error(`Admin Log (${type}): ${message}`, { fileName, recordId, patientId, uploadedBy });
    if (derivedEncryptionKey) {
        encryptAndSaveAllData().catch(e => console.error("Failed to auto-save admin error log:", e));
    }
}

function _internalGenerateNewMedicationEntryHtmlString(type: 'Antibiotics' | 'Analgesics' | 'OtherMedications'): string {
    return `
        <div class="medication-entry" data-med-id="${uuid()}">
            <button type="button" class="remove-medication-btn" data-med-type="${type}" aria-label="Remove medication">&times;</button>
            <div class="form-grid grid-cols-4">
                <div class="input-group"><label>Drug Name</label><input type="text" name="${type.toLowerCase()}-drugName" required></div>
                <div class="input-group"><label>Dose</label><input type="text" name="${type.toLowerCase()}-dose" required></div>
                <div class="input-group"><label>Route</label><input type="text" name="${type.toLowerCase()}-route" required></div>
                <div class="input-group"><label>Frequency</label><input type="text" name="${type.toLowerCase()}-frequency" required></div>
            </div>
            <div class="input-group"><label>Duration (Optional)</label><input type="text" name="${type.toLowerCase()}-duration"></div>
        </div>
    `;
}

function generateNewUniqueKahsPatientId(): string {
    let newId = '';
    let isUnique = false;
    while (!isUnique) {
        // Generate a random 6-digit number, pad with leading zeros if necessary
        const randomSuffix = String(Math.floor(100000 + Math.random() * 900000));
        newId = `KAHS${randomSuffix}`;
        if (!traumaRecords.some(r => r.patientId === newId)) {
            isUnique = true;
        }
    }
    return newId;
}
// --- END Utility Functions ---

// --- START Authentication ---
async function handleLogin(event: Event) {
    event.preventDefault();
    const usernameInput = S<HTMLInputElement>('#username');
    const passwordInput = S<HTMLInputElement>('#password');
    if (!usernameInput || !passwordInput) return;

    const username = usernameInput.value;
    const password = passwordInput.value;
    const user = users.find(u => u.username === username && u.isActive);

    if (user && verifyPassword(password, user.passwordHash)) {
        currentUser = { ...user };

        S<HTMLDivElement>('#login-view')!.style.display = 'none';

        if (currentUser.role === 'admin' || currentUser.role === 'user') {
            S<HTMLDivElement>('#main-app-view')!.style.display = 'flex';
            S<HTMLDivElement>('#main-app-view')!.style.flexDirection = 'column';
            S<HTMLSpanElement>('#welcome-message')!.textContent = `Welcome, ${currentUser.username}! (${currentUser.role})`;

            S<HTMLElement>('#admin-dashboard-link')!.style.display = currentUser.role === 'admin' ? 'flex' : 'none';
            S<HTMLElement>('#ocr-error-log-link')!.style.display = currentUser.role === 'admin' ? 'flex' : 'none';
            const treatmentNavLink = S<HTMLAnchorElement>('#treatment-investigations-link');
            if (treatmentNavLink) treatmentNavLink.style.display = currentUser.role === 'admin' ? 'flex' : 'none';

            if (currentUser.role === 'admin') {
                try {
                    showToast("Admin login successful. Deriving encryption key...", "info");
                    derivedEncryptionKey = await deriveKey(currentUser.username, APP_SECRET_SALT);
                    await loadAndDecryptAllData();
                } catch (kdfError) {
                    console.error("Key derivation failed:", kdfError);
                    showToast("Critical Error: Failed to derive encryption key. Data cannot be secured.", "error");
                    derivedEncryptionKey = null;
                }
            } else if (currentUser.role === 'user') { // Regular 'user'
                 if (!derivedEncryptionKey) {
                    showToast("User login successful. WARNING: Admin has not logged in this session. Data cannot be saved or loaded securely.", "warning");
                } else {
                     showToast("User login successful. Secure session active.", "success");
                }
            }
            navigateTo(currentUser.role === 'admin' ? 'admin-dashboard' : 'patient-details'); // Admin to dashboard, user to patient details


        } else if (currentUser.role === 'nurse') {
            S<HTMLDivElement>('#nurse-upload-view')!.style.display = 'flex';
            S<HTMLDivElement>('#nurse-upload-view')!.style.flexDirection = 'column';
            S<HTMLSpanElement>('#nurse-welcome-message')!.textContent = `Welcome, ${currentUser.username}! (NURSE / DATA ENTRY PERSONNEL)`;
            S<HTMLParagraphElement>('#nurse-role-display')!.textContent = `NURSE / DATA ENTRY PERSONNEL`;


            if (!derivedEncryptionKey) {
                 showToast("Nurse login successful. WARNING: Secure session not fully established by an Admin. Data saving might be limited or insecure.", "warning");
            } else {
                 showToast("Nurse login successful. Secure session active.", "success");
            }
            currentNurseView = 'patientDocUpload'; // Default to patient doc upload
            navigateToNurseView(currentNurseView); // Use new navigation for nurse
        }

    } else {
        showToast('Invalid username or password.', 'error');
    }
}

function handleLogout() {
    currentUser = null;
    editingRecordId = null;
    ocrErrorToRetry = null;
    currentPatientForTreatment = null;
    currentNewlyAddedLabReports = [];
    patientDocumentFile = null;
    patientDocumentBase64 = null;
    extractedBeemaNumber = null;
    currentClinicalImagesData = [];
    currentXRayImagesData = [];
    currentActivePatientIdForNurse = null;
    nurseLabPatientSearchResults = [];
    currentNurseView = 'patientDocUpload';


    derivedEncryptionKey = null;
    // Do not reset users, traumaRecords, adminErrorLog here if admin logged out and user/nurse is still active.
    // Reset them only if the last active user logs out or for full app reset.
    // For simplicity now, let's keep the current data in memory but it won't be saved if no admin key.
    // users = [...initialUsers];
    // traumaRecords = [];
    // adminErrorLog = [];


    S<HTMLDivElement>('#login-view')!.style.display = 'flex';
    S<HTMLDivElement>('#main-app-view')!.style.display = 'none';
    S<HTMLDivElement>('#nurse-upload-view')!.style.display = 'none';
    S<HTMLFormElement>('#login-form')?.reset();
    showToast("Logged out successfully.", "info");
}
// --- END Authentication ---


// --- START Navigation ---
function navigateTo(viewId: string, recordId?: string, errorLogEntry?: AdminErrorLogEntry) {
    if (!currentUser || currentUser.role === 'nurse') { // Nurses use navigateToNurseView
        if (currentUser?.role === 'nurse') {
            navigateToNurseView(currentNurseView); // Default to their current or main view
        } else {
            handleLogout();
        }
        return;
    }

    // Access control for admin/user roles
    const restrictedForUser = ['admin-dashboard', 'ocr-error-log', 'treatment-investigations'];
    if (currentUser.role === 'user' && restrictedForUser.includes(viewId)) {
        showToast("Access Denied. This section is for Admins only.", "error");
        navigateTo('patient-details'); // Redirect user to their default view
        return;
    }
    if ((viewId === 'admin-dashboard' || viewId === 'ocr-error-log' || viewId === 'treatment-investigations') && currentUser.role !== 'admin') {
         showToast("Access Denied. This section is for Admins only.", "error");
         navigateTo('patient-details'); // Redirect user to their default view
        return;
    }

    currentView = viewId; // For main app view state

    const appContent = S<HTMLElement>('#app-content')!;
    appContent.innerHTML = ''; // Clear previous content

    SAll<HTMLAnchorElement>('#app-nav a.nav-link').forEach(link => {
        link.classList.remove('active-link');
        if (link.getAttribute('data-view') === viewId) {
            link.classList.add('active-link');
        }
    });

    editingRecordId = recordId || null;
    ocrErrorToRetry = errorLogEntry || null;
    if (currentView !== 'patient-details') currentNewlyAddedLabReports = [];


    switch (viewId) {
        case 'patient-details': appContent.innerHTML = renderPatientDetailsForm(); attachPatientDetailsFormListeners(editingRecordId); break;
        case 'treatment-investigations': appContent.innerHTML = renderTreatmentInvestigationsView(); attachTreatmentInvestigationsListeners(); break;
        case 'admin-dashboard': appContent.innerHTML = renderAdminDashboard(); attachAdminDashboardListeners(); break;
        case 'ocr-error-log': appContent.innerHTML = renderOcrErrorLogView(); attachOcrErrorLogListeners(); break;
        case 'view-records': renderRecordsListView(); break; // This now renders into #app-content by default
        default: appContent.innerHTML = `<p>View not found.</p>`;
    }
}


function navigateToNurseView(nurseViewId: 'patientDocUpload' | 'labReportUpload' | 'viewLabHistory') {
    if (!currentUser || currentUser.role !== 'nurse') {
        handleLogout();
        return;
    }
    currentNurseView = nurseViewId;
    const nurseAppContent = S<HTMLElement>('#nurse-app-content')!;
    nurseAppContent.innerHTML = ''; // Clear previous content

    SAll<HTMLAnchorElement>('#nurse-app-nav a.nurse-nav-link').forEach(link => {
        link.classList.remove('active-link');
        if (link.getAttribute('data-nurse-view') === nurseViewId) {
            link.classList.add('active-link');
        }
    });

    switch (nurseViewId) {
        case 'patientDocUpload':
            nurseAppContent.innerHTML = renderNursePatientDocUploadSection();
            attachNursePatientDocUploadListeners();
            break;
        case 'labReportUpload':
            nurseAppContent.innerHTML = renderNurseLabReportUploadSection();
            attachNurseLabReportUploadListeners();
            break;
        case 'viewLabHistory':
            nurseAppContent.innerHTML = renderNurseViewLabHistorySection();
            attachNurseViewLabHistoryListeners();
            break;
        default:
            nurseAppContent.innerHTML = `<p>Nurse view not found.</p>`;
    }
}
// --- END Navigation ---

// --- START Patient Details Form ---
let patientDocumentFile: File | null = null;
let patientDocumentBase64: string | null = null;
let extractedBeemaNumber: string | null = null;


let currentClinicalImagesData: Array<{ id: string, dataUrl: string, fileName?: string }> = [];
let currentXRayImagesData: Array<{ id: string, dataUrl: string, fileName?: string }> = [];
let currentNewlyAddedLabReports: LabReportEntry[] = [];


function renderPatientDetailsForm(): string {
    const recordToEdit = editingRecordId ? traumaRecords.find(r => r.id === editingRecordId) : null;
    const todayAD = new Date().toISOString().split('T')[0];
    const todayBS = adbs.ad2bs(todayAD);
    const patientIdGridClass = "form-grid grid-cols-custom-id";

    let patientInfoDisplayValue = recordToEdit ? `Record ID: ${recordToEdit.id}` : 'New Record';
    if (recordToEdit) {
        patientInfoDisplayValue = `Patient ID: ${recordToEdit.patientId}`;
        if (recordToEdit.beemaNumber) {
            patientInfoDisplayValue += ` | Beema No.: ${recordToEdit.beemaNumber}`;
        }
         patientInfoDisplayValue += ` (Record ID: ${recordToEdit.id})`;
    } else if (extractedBeemaNumber) {
         patientInfoDisplayValue = `Patient ID: [Pending] | Beema No.: ${extractedBeemaNumber}`;
    }

    const ageYearsVal = recordToEdit?.age?.years !== undefined ? String(recordToEdit.age.years) : '';
    const ageMonthsVal = recordToEdit?.age?.months !== undefined ? String(recordToEdit.age.months) : '';
    const ageDaysVal = recordToEdit?.age?.days !== undefined ? String(recordToEdit.age.days) : '';

    const labReportUploadSection = `
        <div class="content-card">
            <h3><i class="fas fa-flask"></i> Upload Lab Reports</h3>
            <p class="instruction-text">
                Upload lab report files (PDF, JPG, PNG). OCR will attempt to extract data.
                Files >5MB will be compressed (images only). Admins can manage these in the Treatment tab.
            </p>
            <input type="file" id="patientFormLabReportInput" multiple accept=".pdf,.jpg,.jpeg,.png" style="display:none;">
            <button type="button" id="patientFormUploadLabReportBtn" class="btn btn-secondary btn-sm">
                <i class="fas fa-upload"></i> Upload Lab Report(s)
            </button>
            <div id="lab-ocr-indicator" class="ocr-indicator" style="display: none;">
                <div class="spinner spinner-inline"></div>
                <span>Processing Lab Report OCR...</span>
            </div>
            <div id="patient-form-lab-reports-pending-list" class="mt-2">
            </div>
        </div>
    `;


    return `
        <form id="patient-details-form" class="content-card">
            <h2><i class="fas fa-user-plus"></i> ${recordToEdit ? 'Edit Patient Record' : 'Add New Patient Record'}</h2>

            <div class="content-card">
                <h3><i class="fas fa-id-card"></i> Patient Identification</h3>
                <div class="${patientIdGridClass}">
                    <div class="input-group">
                        <label for="patientIdInternal">Patient ID (KAHS...) <span class="text-danger">*</span></label>
                        <input type="text" id="patientIdInternal" name="patientIdInternal" required
                               value="${recordToEdit?.patientId || ''}"
                               pattern="^KAHS\\d+$" title="Patient ID must start with KAHS followed by numbers (e.g., KAHS12345)."
                               placeholder="KAHS12345">
                    </div>
                    <div class="input-group">
                        <label for="patientName">Patient Full Name <span class="text-danger">*</span></label>
                        <input type="text" id="patientName" name="patientName" required value="${recordToEdit?.patientName || ''}">
                    </div>
                     <div class="input-group">
                        <label for="ageYears">Age (Years) <span class="text-danger">*</span></label>
                        <input type="number" id="ageYears" name="ageYears" required value="${ageYearsVal}" min="0" placeholder="Years">
                    </div>
                    <div class="input-group">
                        <label for="ageMonths">Age (Months)</label>
                        <input type="number" id="ageMonths" name="ageMonths" value="${ageMonthsVal}" min="0" max="11" placeholder="Months">
                    </div>
                    <div class="input-group">
                        <label for="ageDays">Age (Days)</label>
                        <input type="number" id="ageDays" name="ageDays" value="${ageDaysVal}" min="0" max="31" placeholder="Days">
                    </div>
                    <div class="input-group">
                        <label for="sex">Sex <span class="text-danger">*</span></label>
                        <select id="sex" name="sex" required>
                            <option value="" disabled ${!recordToEdit?.sex ? 'selected' : ''}>Select Sex</option>
                            <option value="Male" ${recordToEdit?.sex === 'Male' ? 'selected' : ''}>Male</option>
                            <option value="Female" ${recordToEdit?.sex === 'Female' ? 'selected' : ''}>Female</option>
                            <option value="Other" ${recordToEdit?.sex === 'Other' ? 'selected' : ''}>Other</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label for="address">Address</label>
                        <input type="text" id="address" name="address" value="${recordToEdit?.address || ''}">
                    </div>
                    <div class="input-group">
                        <label for="contactNumber">Contact Number</label>
                        <input type="tel" id="contactNumber" name="contactNumber" value="${recordToEdit?.contactNumber || ''}"
                               placeholder="+97798XXXXXXXX" pattern="^\\+977\\d{9,10}$"
                               title="Contact number must start with +977 followed by 9 or 10 digits (e.g., +9779812345678).">
                    </div>
                </div>
                <div class="input-group mt-2">
                    <label for="beemaNumberDisplay">Beema Number (From OCR)</label>
                    <input type="text" id="beemaNumberDisplay" name="beemaNumberDisplay" readonly
                           value="${recordToEdit?.beemaNumber || extractedBeemaNumber || 'Not extracted'}"
                           style="${(recordToEdit?.beemaNumber || extractedBeemaNumber) ? '' : 'display:block;'}">
                </div>
                <div class="input-group mt-2">
                    <label for="recordIdDisplay">Record Identifier</label>
                    <input type="text" id="recordIdDisplay" name="recordIdDisplay" readonly value="${patientInfoDisplayValue}">
                </div>
            </div>

            <div class="content-card">
                <h3><i class="fas fa-info-circle"></i> Additional Details</h3>
                <p>[Insert relevant details here]</p>
            </div>

            <div id="patient-document-section" class="content-card">
                <h3><i class="fas fa-file-alt"></i> Patient Document Scan (ID/Referral/Beema Card)</h3>
                <p class="instruction-text">
                    Upload or capture an image of the patient's identification document, referral letter, or Beema Insurance Card.
                    OCR will attempt to extract details including Beema No., detailed Age (Y/M/D or from DOB BS), and 10-digit Contact. Max 5MB (larger files will be compressed). PDFs are accepted.
                </p>
                <div class="document-actions">
                    <input type="file" id="patientDocumentInput" accept="image/jpeg,image/png,image/gif,application/pdf" style="display: none;">
                    <button type="button" id="uploadPatientDocumentBtn" class="btn btn-secondary"><i class="fas fa-upload"></i> Upload Document</button>
                    <button type="button" id="capturePatientDocumentBtn" class="btn btn-secondary"><i class="fas fa-camera"></i> Capture with Camera</button>
                </div>
                <div id="patient-doc-ocr-indicator" class="ocr-indicator" style="display: none;">
                    <div class="spinner spinner-inline"></div>
                    <span>Processing OCR... please wait.</span>
                </div>
                <div id="patient-document-preview-area">
                    ${recordToEdit?.patientDocumentUrl ? (recordToEdit.patientDocumentUrl.startsWith('data:application/pdf') ? '<i class="fas fa-file-pdf" style="font-size: 5rem; color: var(--danger-color);"></i><p>PDF Document Uploaded</p>' : `<img src="${recordToEdit.patientDocumentUrl}" alt="Patient Document Preview">`) : 'No document uploaded/captured yet.'}
                </div>
                ${recordToEdit?.patientDocumentUrl ? `
                    <button type="button" id="clearPatientDocumentBtn" class="btn btn-danger-outline btn-sm mt-2">
                        <i class="fas fa-trash-alt"></i> Clear Scanned Document
                    </button>` : ''}
                <div class="input-group mt-2">
                    <label for="patientDocumentOcrText">Extracted OCR Text (Review/Edit)</label>
                    <textarea id="patientDocumentOcrText" name="patientDocumentOcrText" rows="3" placeholder="OCR text will appear here...">${recordToEdit?.patientDocumentOcrText || ''}</textarea>
                </div>
            </div>

            <div class="content-card">
                <h3><i class="fas fa-comment-medical"></i> Chief Complaints</h3>
                 <div class="input-group">
                    <label for="chiefComplaints">Chief Complaints (Max 500 chars)</label>
                    <textarea id="chiefComplaints" name="chiefComplaints" rows="3" maxlength="500" placeholder="e.g., Pain and swelling in the right ankle">${recordToEdit?.chiefComplaints || ''}</textarea>
                </div>
            </div>

            <div class="content-card">
                <h3><i class="fas fa-user-md"></i> Injury Details</h3>
                <div class="form-grid grid-cols-3">
                    <div class="input-group">
                        <label for="dateOfInjuryAD">Date of Injury (AD) <span class="text-danger">*</span></label>
                        <input type="date" id="dateOfInjuryAD" name="dateOfInjuryAD" required value="${recordToEdit?.dateOfInjuryAD || todayAD}">
                    </div>
                    <div class="input-group">
                        <label for="dateOfInjuryBS">Date of Injury (BS)</label>
                        <input type="text" id="dateOfInjuryBS" name="dateOfInjuryBS" readonly value="${recordToEdit?.dateOfInjuryBS || todayBS}">
                    </div>
                    <div class="input-group">
                        <label for="timeOfInjury">Time of Injury <span class="text-danger">*</span></label>
                        <input type="time" id="timeOfInjury" name="timeOfInjury" required value="${recordToEdit?.timeOfInjury || ''}">
                    </div>
                    <div class="input-group">
                        <label for="modeOfInjury">Mechanism of Injury <span class="text-danger">*</span></label>
                        <select id="modeOfInjury" name="modeOfInjury" required>
                            <option value="" ${!recordToEdit?.modeOfInjury ? 'selected' : ''} disabled>Select Mechanism</option>
                            <option value="Falls" ${recordToEdit?.modeOfInjury === 'Falls' ? 'selected' : ''}>Falls (including falls from height)</option>
                            <option value="Road traffic accidents" ${recordToEdit?.modeOfInjury === 'Road traffic accidents' ? 'selected' : ''}>Road traffic accidents</option>
                            <option value="Physical assault" ${recordToEdit?.modeOfInjury === 'Physical assault' ? 'selected' : ''}>Physical assault</option>
                            <option value="Burns" ${recordToEdit?.modeOfInjury === 'Burns' ? 'selected' : ''}>Burns</option>
                            <option value="Sports-related injuries" ${recordToEdit?.modeOfInjury === 'Sports-related injuries' ? 'selected' : ''}>Sports-related injuries</option>
                            <option value="Occupational injuries" ${recordToEdit?.modeOfInjury === 'Occupational injuries' ? 'selected' : ''}>Occupational injuries</option>
                            <option value="Domestic injuries and cuts" ${recordToEdit?.modeOfInjury === 'Domestic injuries and cuts' ? 'selected' : ''}>Domestic injuries and cuts</option>
                            <option value="Self-harm" ${recordToEdit?.modeOfInjury === 'Self-harm' ? 'selected' : ''}>Self-harm (intentional poisoning or injury)</option>
                            <option value="Crush injury" ${recordToEdit?.modeOfInjury === 'Crush injury' ? 'selected' : ''}>Crush injury</option>
                            <option value="Twisting of ankle" ${recordToEdit?.modeOfInjury === 'Twisting of ankle' ? 'selected' : ''}>Twisting of ankle</option>
                            <option value="Twisting of leg" ${recordToEdit?.modeOfInjury === 'Twisting of leg' ? 'selected' : ''}>Twisting of leg</option>
                            <option value="Other" ${recordToEdit?.modeOfInjury === 'Other' ? 'selected' : ''}>Other</option>
                        </select>
                    </div>
                    <div class="input-group" id="otherMOI-group" style="${recordToEdit?.modeOfInjury === 'Other' ? '' : 'display: none;'}">
                        <label for="otherMOI">If Other, Specify</label>
                        <input type="text" id="otherMOI" name="otherMOI" value="${recordToEdit?.otherMOI || ''}">
                    </div>
                    <div class="input-group">
                        <label for="siteOfInjury">Site of Injury <span class="text-danger">*</span></label>
                        <input type="text" id="siteOfInjury" name="siteOfInjury" required value="${recordToEdit?.siteOfInjury || ''}">
                    </div>
                    <div class="input-group">
                        <label for="typeOfInjury">Type of Injury <span class="text-danger">*</span></label>
                        <input type="text" id="typeOfInjury" name="typeOfInjury" required value="${recordToEdit?.typeOfInjury || ''}">
                    </div>
                </div>
                <div class="input-group">
                    <label for="descriptionOfInjuries">Description of Injuries</label>
                    <textarea id="descriptionOfInjuries" name="descriptionOfInjuries" rows="4" maxlength="1000" placeholder="Detailed description of the injuries sustained...">${recordToEdit?.descriptionOfInjuries || ''}</textarea>
                </div>
            </div>

            <div class="content-card">
                <h2><i class="fas fa-x-ray"></i> X-ray Upload</h2>
                <p class="instruction-text">
                    Upload up to ${MAX_XRAY_IMAGES} X-ray images (JPG, JPEG, PNG). Max file size: ${MAX_XRAY_FILE_SIZE_BYTES / (1024 * 1024)}MB each (after any initial compression for files >5MB).
                    Images will be compressed if larger than ${XRAY_IMAGE_MAX_DIMENSION}px.
                </p>
                <div id="xray-images-gallery" class="image-gallery-grid">
                </div>
                <input type="file" id="xrayImageInput" accept=".jpg,.jpeg,.png" multiple style="display: none;" aria-label="X-ray image uploader">
                <p id="xray-images-helper-text" class="mt-1 text-muted-color image-gallery-helper-text">0/${MAX_XRAY_IMAGES} images uploaded.</p>
            </div>

            ${labReportUploadSection}

            <div class="content-card">
                <h3><i class="fas fa-diagnoses"></i> Presentation & Examination / Diagnosis</h3>
                <div class="form-grid grid-cols-2">
                    <div class="input-group">
                        <label for="presentComplaint">Present Complaint <span class="text-danger">*</span></label>
                        <input type="text" id="presentComplaint" name="presentComplaint" required value="${recordToEdit?.presentComplaint || ''}">
                    </div>
                    <div class="input-group">
                        <label for="glasgowComaScale">Glasgow Coma Scale <span class="text-danger">*</span></label>
                        <input type="text" id="glasgowComaScale" name="glasgowComaScale" required value="${recordToEdit?.glasgowComaScale || ''}">
                    </div>
                </div>
                <div class="input-group">
                    <label for="vitalSigns">Vital Signs <span class="text-danger">*</span></label>
                    <textarea id="vitalSigns" name="vitalSigns" rows="3" required placeholder="e.g., BP: 120/80 mmHg, HR: 75 bpm, RR: 16/min, Temp: 98.6°F">${recordToEdit?.vitalSigns || ''}</textarea>
                </div>
                <div class="form-grid grid-cols-2">
                    <div class="input-group">
                        <label for="systemicExamination">Systemic Examination <span class="text-danger">*</span></label>
                        <textarea id="systemicExamination" name="systemicExamination" rows="3" required>${recordToEdit?.systemicExamination || ''}</textarea>
                    </div>
                    <div class="input-group">
                        <label for="localExamination">Local Examination <span class="text-danger">*</span></label>
                        <textarea id="localExamination" name="localExamination" rows="3" required>${recordToEdit?.localExamination || ''}</textarea>
                    </div>
                </div>
                 <div class="input-group">
                    <label>Diagnosis Side (if applicable)</label>
                    <div>
                        <input type="checkbox" id="diagnosisSideRight" name="diagnosisSide" value="Right" ${recordToEdit?.diagnosisSide?.includes('Right') ? 'checked' : ''}>
                        <label for="diagnosisSideRight" style="font-weight: normal; margin-right: 15px;">Right</label>
                        <input type="checkbox" id="diagnosisSideLeft" name="diagnosisSide" value="Left" ${recordToEdit?.diagnosisSide?.includes('Left') ? 'checked' : ''}>
                        <label for="diagnosisSideLeft" style="font-weight: normal;">Left</label>
                    </div>
                </div>
                <div class="input-group">
                    <label for="provisionalDiagnosis">Provisional Diagnosis <span class="text-danger">*</span></label>
                    <input type="text" id="provisionalDiagnosis" name="provisionalDiagnosis" required value="${recordToEdit?.provisionalDiagnosis || ''}">
                </div>
            </div>

            <div class="content-card clinical-images-section">
                <h3><i class="fas fa-images"></i> Clinical Images</h3>
                <p class="instruction-text">
                    Upload up to ${MAX_CLINICAL_IMAGES} clinical images (e.g., wounds, deformities). Files >5MB will be initially compressed.
                </p>
                <div id="clinical-images-gallery" class="image-gallery-grid">
                </div>
                <input type="file" id="clinicalImageInput" accept="image/jpeg,image/png" multiple style="display: none;" aria-label="Clinical image uploader">
                <button type="button" id="captureClinicalImageBtn" class="btn btn-secondary btn-sm mt-2"><i class="fas fa-camera"></i> Capture Clinical Image</button>
                <p id="clinical-images-helper-text" class="mt-1 text-muted-color image-gallery-helper-text">0/${MAX_CLINICAL_IMAGES} images uploaded.</p>
            </div>

            <div class="form-actions">
                <button type="button" id="clear-form-btn" class="btn btn-secondary">Clear Form</button>
                <button type="submit" class="btn btn-primary">${recordToEdit ? '<i class="fas fa-save"></i> Update Record' : '<i class="fas fa-plus-circle"></i> Add Record'}</button>
            </div>
        </form>
    `;
}

function attachPatientDetailsFormListeners(recordIdForEdit: string | null = null) {
    const form = S<HTMLFormElement>('#patient-details-form');
    if (!form) return;

    const recordToEdit = recordIdForEdit ? traumaRecords.find(r => r.id === recordIdForEdit) : null;
    extractedBeemaNumber = recordToEdit?.beemaNumber || null;
    currentNewlyAddedLabReports = [];
    renderPendingLabReportsList();


    if (recordToEdit && recordToEdit.clinicalImageUrls) {
        currentClinicalImagesData = recordToEdit.clinicalImageUrls.map(url => ({ id: uuid(), dataUrl: url, fileName: 'Stored Image' }));
    } else {
        currentClinicalImagesData = [];
    }
    renderClinicalImageGalleryUI();

    if (recordToEdit && recordToEdit.xrayImageUrls) {
        currentXRayImagesData = recordToEdit.xrayImageUrls.map(url => ({ id: uuid(), dataUrl: url, fileName: 'Stored X-ray' }));
    } else {
        currentXRayImagesData = [];
    }
    renderXRayImageGalleryUI();

    const patientDocumentInput = S<HTMLInputElement>('#patientDocumentInput');
    const uploadPatientDocumentBtn = S<HTMLButtonElement>('#uploadPatientDocumentBtn');
    const capturePatientDocumentBtn = S<HTMLButtonElement>('#capturePatientDocumentBtn');
    const patientDocumentPreviewArea = S<HTMLDivElement>('#patient-document-preview-area');
    const patientDocumentOcrText = S<HTMLTextAreaElement>('#patientDocumentOcrText');
    const clearPatientDocumentBtn = S<HTMLButtonElement>('#clearPatientDocumentBtn');
    const beemaNumberDisplayEl = S<HTMLInputElement>('#beemaNumberDisplay');
    const recordIdDisplayEl = S<HTMLInputElement>('#recordIdDisplay');


    uploadPatientDocumentBtn?.addEventListener('click', () => patientDocumentInput?.click());

    patientDocumentInput?.addEventListener('change', async (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (files && files[0]) {
            let fileToProcess = files[0];
            let isPdf = fileToProcess.type === 'application/pdf';


            if (fileToProcess.size > LARGE_FILE_THRESHOLD_BYTES && !isPdf) {
                showToast("Patient document image is large, attempting compression...", "info");
                try {
                    const compressedBase64 = await compressImage(fileToProcess, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_QUALITY);
                    fileToProcess = await base64StringToFile(compressedBase64, fileToProcess.name, fileToProcess.type);
                    showToast("Compression successful for patient document image.", "success");
                } catch (compressionError) {
                    console.error("Error compressing patient document image:", compressionError);
                    showToast("Failed to compress large patient document image. Proceeding with original.", "warning");
                }
            } else if (isPdf && fileToProcess.size > 10 * 1024 * 1024) {
                 showToast("PDF Patient document is very large (max 10MB). Processing might be slow or fail.", "warning");
            }


             if (fileToProcess.size > 10 * 1024 * 1024) {
                showToast("Patient document file still too large after any compression (max 10MB for processing).", "error");
                return;
            }

            patientDocumentFile = fileToProcess;
            patientDocumentBase64 = await fileToDataURL(fileToProcess);


            if (patientDocumentPreviewArea) {
                if (isPdf) {
                    patientDocumentPreviewArea.innerHTML = '<i class="fas fa-file-pdf" style="font-size: 5rem; color: var(--danger-color);"></i><p>PDF Document Uploaded</p>';
                } else {
                    patientDocumentPreviewArea.innerHTML = `<img src="${patientDocumentBase64}" alt="Patient Document Preview">`;
                }
                const existingClearBtn = S<HTMLButtonElement>('#clearPatientDocumentBtn');
                if (!existingClearBtn) {
                    const newClearBtn = document.createElement('button');
                    newClearBtn.type = 'button';
                    newClearBtn.id = 'clearPatientDocumentBtn';
                    newClearBtn.className = 'btn btn-danger-outline btn-sm mt-2';
                    newClearBtn.innerHTML = '<i class="fas fa-trash-alt"></i> Clear Scanned Document';
                    newClearBtn.addEventListener('click', handleClearPatientDocument);
                    patientDocumentPreviewArea.insertAdjacentElement('afterend', newClearBtn);
                } else {
                     existingClearBtn.style.display = 'block';
                }
            }
            const ocrResult = await performOcr(fileToProcess, 'patient_document');
            if (ocrResult.rawText && patientDocumentOcrText) {
                patientDocumentOcrText.value = ocrResult.rawText;
                if (!ocrResult.error) showToast("OCR processing complete.", "success");
                else showToast(`OCR Information: ${ocrResult.error}`, "info");


                const demographics = ocrResult.extractedDemographics;
                if (demographics) {
                    if (demographics.patientName && S<HTMLInputElement>('#patientName')) {
                        (S<HTMLInputElement>('#patientName')!).value = demographics.patientName;
                    }
                    if (demographics.patientId && S<HTMLInputElement>('#patientIdInternal')) {
                        (S<HTMLInputElement>('#patientIdInternal')!).value = demographics.patientId;
                    }
                    const ageYearsEl = S<HTMLInputElement>('#ageYears');
                    const ageMonthsEl = S<HTMLInputElement>('#ageMonths');
                    const ageDaysEl = S<HTMLInputElement>('#ageDays');

                    if (demographics.age && typeof demographics.age.years === 'number') { // Check years specifically
                        if (ageYearsEl) ageYearsEl.value = String(demographics.age.years);
                        if (ageMonthsEl) ageMonthsEl.value = demographics.age.months !== undefined ? String(demographics.age.months) : '';
                        if (ageDaysEl) ageDaysEl.value = demographics.age.days !== undefined ? String(demographics.age.days) : '';
                    } else if (demographics.ageYears !== undefined) {
                        if (ageYearsEl) ageYearsEl.value = String(demographics.ageYears);
                        if (ageMonthsEl) ageMonthsEl.value = demographics.ageMonths !== undefined ? String(demographics.ageMonths) : '';
                        if (ageDaysEl) ageDaysEl.value = demographics.ageDays !== undefined ? String(demographics.ageDays) : '';
                    }


                    if (demographics.sex && S<HTMLSelectElement>('#sex')) {
                        (S<HTMLSelectElement>('#sex')!).value = demographics.sex;
                    }
                    if (demographics.address && S<HTMLInputElement>('#address')) {
                        (S<HTMLInputElement>('#address')!).value = demographics.address;
                    }
                    if (demographics.contactNumber && S<HTMLInputElement>('#contactNumber')) {
                        (S<HTMLInputElement>('#contactNumber')!).value = demographics.contactNumber;
                    }
                    extractedBeemaNumber = demographics.beemaNumber || null;
                    if (beemaNumberDisplayEl) {
                        if (extractedBeemaNumber) {
                            beemaNumberDisplayEl.value = extractedBeemaNumber;
                            beemaNumberDisplayEl.style.display = 'block';
                        } else {
                            beemaNumberDisplayEl.value = 'Not extracted';
                            beemaNumberDisplayEl.style.display = 'block';
                        }
                    }
                    updateRecordIdDisplay();
                    if (Object.keys(demographics).length > 0) {
                       showToast("Demographics partially filled from OCR.", "info");
                    }
                }
            } else if (ocrResult.error) {
                showToast(`OCR Error: ${ocrResult.error}`, "error");
                if (patientDocumentOcrText) patientDocumentOcrText.value = `OCR Failed: ${ocrResult.error}`;
                 extractedBeemaNumber = null;
                if(beemaNumberDisplayEl) {
                    beemaNumberDisplayEl.value = 'Not extracted';
                    beemaNumberDisplayEl.style.display = 'block';
                }
                updateRecordIdDisplay();
            }
        }
    });

    function updateRecordIdDisplay() {
        const currentPatientId = S<HTMLInputElement>('#patientIdInternal')?.value || (recordToEdit ? recordToEdit.patientId : '[Pending]');
        if (recordIdDisplayEl) {
            let displayVal = `Patient ID: ${currentPatientId}`;
            if (extractedBeemaNumber) {
                displayVal += ` | Beema No.: ${extractedBeemaNumber}`;
            }
            if (recordToEdit) {
                 displayVal += ` (Record ID: ${recordToEdit.id})`;
            } else if (!recordToEdit && !currentPatientId.startsWith("KAHS")) {
                 displayVal = "New Record";
                 if(extractedBeemaNumber) displayVal += ` | Beema No.: ${extractedBeemaNumber}`;
            }
            recordIdDisplayEl.value = displayVal;
        }
    }


    function handleClearPatientDocument() {
        patientDocumentFile = null;
        patientDocumentBase64 = null;
        extractedBeemaNumber = null;
        if (patientDocumentInput) patientDocumentInput.value = '';
        if (patientDocumentPreviewArea) patientDocumentPreviewArea.innerHTML = 'No document uploaded/captured yet.';
        if (patientDocumentOcrText) patientDocumentOcrText.value = '';
        if (beemaNumberDisplayEl) {
            beemaNumberDisplayEl.value = 'Not extracted';
            beemaNumberDisplayEl.style.display = 'block';
        }
        updateRecordIdDisplay();
        const btnToClear = S<HTMLButtonElement>('#clearPatientDocumentBtn');
        if (btnToClear) btnToClear.remove();
    }

    if (clearPatientDocumentBtn) {
        clearPatientDocumentBtn.addEventListener('click', handleClearPatientDocument);
        if (!recordToEdit?.patientDocumentUrl) clearPatientDocumentBtn.style.display = 'none';
    }
    if (recordToEdit?.patientDocumentUrl && patientDocumentPreviewArea) {
        patientDocumentBase64 = recordToEdit.patientDocumentUrl;
    }
    if (beemaNumberDisplayEl) {
        beemaNumberDisplayEl.value = recordToEdit?.beemaNumber || extractedBeemaNumber || 'Not extracted';
        beemaNumberDisplayEl.style.display = 'block';
    }
    updateRecordIdDisplay();

    capturePatientDocumentBtn?.addEventListener('click', () => {
        openCameraModal('patientDocument');
    });

    const dateOfInjuryADInput = S<HTMLInputElement>('#dateOfInjuryAD');
    const dateOfInjuryBSInput = S<HTMLInputElement>('#dateOfInjuryBS');
    dateOfInjuryADInput?.addEventListener('change', () => {
        if (dateOfInjuryADInput.value && dateOfInjuryBSInput) {
            dateOfInjuryBSInput.value = adbs.ad2bs(dateOfInjuryADInput.value);
        }
    });

    const modeOfInjurySelect = S<HTMLSelectElement>('#modeOfInjury');
    const otherMOIGroup = S<HTMLDivElement>('#otherMOI-group');
    modeOfInjurySelect?.addEventListener('change', () => {
        if (modeOfInjurySelect.value === 'Other') {
            if (otherMOIGroup) otherMOIGroup.style.display = 'block';
        } else {
            if (otherMOIGroup) otherMOIGroup.style.display = 'none';
            const otherMOIInput = S<HTMLInputElement>('#otherMOI');
            if(otherMOIInput) otherMOIInput.value = '';
        }
    });

    const clinicalImageInput = S<HTMLInputElement>('#clinicalImageInput');
    const clinicalImagesGallery = S<HTMLDivElement>('#clinical-images-gallery');
    const captureClinicalImageBtn = S<HTMLButtonElement>('#captureClinicalImageBtn');

    captureClinicalImageBtn?.addEventListener('click', () => {
        openCameraModal('clinicalImage');
    });

    clinicalImagesGallery?.addEventListener('click', async (event) => {
        const target = event.target as HTMLElement;
        if (target.closest('.gallery-add-card')) {
            clinicalImageInput?.click();
        } else if (target.closest('.delete-image-btn')) {
            const button = target.closest<HTMLButtonElement>('.delete-image-btn');
            const imageId = button?.dataset.id;
            if (imageId) {
                currentClinicalImagesData = currentClinicalImagesData.filter(img => img.id !== imageId);
                renderClinicalImageGalleryUI();
            }
        }
    });

    clinicalImageInput?.addEventListener('change', async (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (!files) return;
        let filesProcessed = 0;
        for (const originalFile of Array.from(files)) {
            if (currentClinicalImagesData.length >= MAX_CLINICAL_IMAGES) {
                showToast(`Cannot upload more than ${MAX_CLINICAL_IMAGES} clinical images.`, 'warning');
                break;
            }
            try {
                let fileToCompress = originalFile;
                if (originalFile.size > LARGE_FILE_THRESHOLD_BYTES) {
                    showToast(`Clinical image ${originalFile.name} is large, attempting initial compression...`, "info");
                    const initialBase64 = await compressImage(originalFile, LARGE_FILE_GENERAL_COMPRESSION_MAX_DIM, LARGE_FILE_GENERAL_COMPRESSION_MAX_DIM, LARGE_FILE_GENERAL_COMPRESSION_QUALITY);
                    fileToCompress = await base64StringToFile(initialBase64, originalFile.name, originalFile.type);
                    showToast(`Initial compression for ${originalFile.name} successful.`, "success");
                }
                const compressedDataUrl = await compressImage(fileToCompress, CLINICAL_IMAGE_MAX_DIMENSION, CLINICAL_IMAGE_MAX_DIMENSION, CLINICAL_IMAGE_COMPRESSION_QUALITY);
                currentClinicalImagesData.push({ id: uuid(), dataUrl: compressedDataUrl, fileName: originalFile.name });
                filesProcessed++;
            } catch (error) {
                console.error("Error compressing clinical image:", error);
                showToast(`Error compressing ${originalFile.name}.`, 'error');
            }
        }
        if (filesProcessed > 0) renderClinicalImageGalleryUI();
        if (clinicalImageInput) clinicalImageInput.value = '';
    });

    const xrayImageInput = S<HTMLInputElement>('#xrayImageInput');
    const xrayImagesGallery = S<HTMLDivElement>('#xray-images-gallery');

    xrayImagesGallery?.addEventListener('click', async (event) => {
        const target = event.target as HTMLElement;
        if (target.closest('.gallery-add-card')) {
            xrayImageInput?.click();
        } else if (target.closest('.delete-image-btn')) {
            const button = target.closest<HTMLButtonElement>('.delete-image-btn');
            const imageId = button?.dataset.id;
            if (imageId) {
                currentXRayImagesData = currentXRayImagesData.filter(img => img.id !== imageId);
                renderXRayImageGalleryUI();
            }
        }
    });

    xrayImageInput?.addEventListener('change', async (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (!files) return;
        let filesProcessed = 0;
        for (const originalFile of Array.from(files)) {
            if (currentXRayImagesData.length >= MAX_XRAY_IMAGES) {
                showToast(`Cannot upload more than ${MAX_XRAY_IMAGES} X-ray images.`, 'warning');
                break;
            }
            const acceptedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
            if (!acceptedTypes.includes(originalFile.type)) {
                 showToast(`Invalid file type for X-ray: ${originalFile.name}. Only JPG, JPEG, PNG allowed.`, 'error');
                 continue;
            }

            try {
                let fileToProcess = originalFile;
                if (originalFile.size > LARGE_FILE_THRESHOLD_BYTES) {
                    showToast(`X-ray ${originalFile.name} is large (>5MB), attempting initial compression...`, "info");
                    const initialBase64 = await compressImage(originalFile, LARGE_FILE_GENERAL_COMPRESSION_MAX_DIM, LARGE_FILE_GENERAL_COMPRESSION_MAX_DIM, LARGE_FILE_GENERAL_COMPRESSION_QUALITY);
                    fileToProcess = await base64StringToFile(initialBase64, originalFile.name, originalFile.type);
                    showToast(`Initial compression for X-ray ${originalFile.name} successful.`, "success");
                }

                if (fileToProcess.size > MAX_XRAY_FILE_SIZE_BYTES) {
                    showToast(`X-ray image "${fileToProcess.name}" is still too large (max ${MAX_XRAY_FILE_SIZE_BYTES / (1024*1024)}MB after initial compression).`, 'error');
                    continue;
                }

                const compressedDataUrl = await compressImage(fileToProcess, XRAY_IMAGE_MAX_DIMENSION, XRAY_IMAGE_MAX_DIMENSION, XRAY_IMAGE_COMPRESSION_QUALITY);
                currentXRayImagesData.push({ id: uuid(), dataUrl: compressedDataUrl, fileName: originalFile.name });
                filesProcessed++;
            } catch (error) {
                console.error("Error processing X-ray image:", error);
                showToast(`Error processing X-ray ${originalFile.name}.`, 'error');
            }
        }
        if (filesProcessed > 0) renderXRayImageGalleryUI();
        if (xrayImageInput) xrayImageInput.value = '';
    });

    const patientIdInputEl = S<HTMLInputElement>('#patientIdInternal');
    patientIdInputEl?.addEventListener('blur', (event) => {
        const input = event.target as HTMLInputElement;
        let value = input.value.trim();
        if (value.length > 0) {
            if (/^\d+$/.test(value)) {
                input.value = "KAHS" + value;
            } else if (!value.toUpperCase().startsWith("KAHS") && value.match(/\d+/) && !value.match(/[a-zA-Z]/)) {
                 input.value = "KAHS" + value.replace(/\D/g, '');
            } else if (value.toUpperCase().startsWith("KAHS")) {
                const numericPart = value.substring(4);
                input.value = "KAHS" + numericPart.replace(/\D/g, '');
            } else if (value.toUpperCase().includes("KAHS")) {
                 const parts = value.toUpperCase().split("KAHS");
                 const numericPart = (parts[1] || parts[0]).replace(/\D/g, '');
                 if (numericPart.length > 0) {
                    input.value = "KAHS" + numericPart;
                 }
            }
        }
        updateRecordIdDisplay();
    });
    patientIdInputEl?.addEventListener('input', updateRecordIdDisplay);

    const patientFormLabReportInput = S<HTMLInputElement>('#patientFormLabReportInput');
    const patientFormUploadLabReportBtn = S<HTMLButtonElement>('#patientFormUploadLabReportBtn');

    patientFormUploadLabReportBtn?.addEventListener('click', () => patientFormLabReportInput?.click());

    patientFormLabReportInput?.addEventListener('change', async (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (!files) return;

        for (const originalFile of Array.from(files)) {
            const tempDisplayId = uuid();
            const labOcrIndicator = S<HTMLDivElement>('#lab-ocr-indicator');
            if(labOcrIndicator) labOcrIndicator.style.display = 'flex';


            const pendingListContainer = S<HTMLDivElement>('#patient-form-lab-reports-pending-list');
            const tempPendingItem = document.createElement('div');
            tempPendingItem.className = 'pending-lab-report-item';
            tempPendingItem.dataset.tempId = tempDisplayId;
            tempPendingItem.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Processing ${originalFile.name}...`;
            pendingListContainer?.appendChild(tempPendingItem);


            let fileForOcr = originalFile;
            let base64Url = '';

            if (originalFile.type.startsWith('image/') && originalFile.size > LARGE_FILE_THRESHOLD_BYTES) {
                showToast(`Lab report image ${originalFile.name} is large, attempting compression...`, "info");
                try {
                    base64Url = await compressImage(originalFile, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_QUALITY);
                    fileForOcr = await base64StringToFile(base64Url, originalFile.name, originalFile.type);
                    showToast(`Compression for ${originalFile.name} successful.`, "success");
                } catch (e) {
                    console.error(`Compression error for ${originalFile.name}:`, e);
                    showToast(`Compression failed for ${originalFile.name}. Using original.`, "warning");
                    base64Url = await fileToDataURL(originalFile);
                }
            } else {
                base64Url = await fileToDataURL(originalFile);
            }

            if ((originalFile.type === "application/pdf" || fileForOcr.type === "application/pdf") && fileForOcr.size > 10 * 1024 * 1024) {
                showToast(`PDF Lab report "${fileForOcr.name}" is very large. OCR might be slow or fail.`, "warning");
            }

            const ocrResult = await performOcr(fileForOcr, 'lab_report');
            if(labOcrIndicator) labOcrIndicator.style.display = 'none';

            const newLabEntry: LabReportEntry = {
                id: uuid(),
                fileName: originalFile.name,
                fileUrl: base64Url,
                fileType: fileForOcr.type,
                reportDate: ocrResult.extractedLabData?.reportDate,
                labParameters: ocrResult.extractedLabData?.labParameters,
                rawOcrText: ocrResult.rawText,
                source: 'auto-captured',
                createdAt: new Date().toISOString()
            };
            currentNewlyAddedLabReports.push(newLabEntry);
            renderPendingLabReportsList();

            const processedTempItem = pendingListContainer?.querySelector(`[data-temp-id="${tempDisplayId}"]`);
            if (processedTempItem && processedTempItem.innerHTML.includes('fa-spinner')) {
                 processedTempItem.remove();
            }


            if (ocrResult.error) {
                showToast(`OCR for ${originalFile.name}: ${ocrResult.error}`, 'warning');
            } else if (!ocrResult.extractedLabData?.reportDate && !ocrResult.extractedLabData?.labParameters) {
                showToast(`OCR for ${originalFile.name} completed, but no structured data (date/parameters) could be extracted. Review raw text.`, 'info');
            } else {
                showToast(`Lab report ${originalFile.name} processed and staged for saving.`, 'success');
            }
        }
        if (patientFormLabReportInput) patientFormLabReportInput.value = '';
    });

    S<HTMLDivElement>('#patient-form-lab-reports-pending-list')?.addEventListener('click', (event) => {
        const target = event.target as HTMLElement;
        const removeBtn = target.closest<HTMLButtonElement>('.remove-pending-lab-btn');
        if (removeBtn) {
            const labEntryIdToRemove = removeBtn.dataset.id;
            if (labEntryIdToRemove) {
                currentNewlyAddedLabReports = currentNewlyAddedLabReports.filter(entry => entry.id !== labEntryIdToRemove);
                renderPendingLabReportsList();
                showToast("Staged lab report removed.", "info");
            }
        }
    });


    form.addEventListener('submit', (event) => handlePatientDetailsFormSubmit(event, recordIdForEdit));

    const clearFormBtn = S<HTMLButtonElement>('#clear-form-btn');
    clearFormBtn?.addEventListener('click', () => {
        form.reset();
        const currentBSDateInput = S<HTMLInputElement>('#dateOfInjuryBS');
        if (currentBSDateInput) currentBSDateInput.value = adbs.ad2bs(new Date().toISOString().split('T')[0]);

        const recordIdDisplayInput = S<HTMLInputElement>('#recordIdDisplay');
        if (recordIdDisplayInput) recordIdDisplayInput.value = 'New Record';
        const patientIdInternalInput = S<HTMLInputElement>('#patientIdInternal');
        if (patientIdInternalInput) patientIdInternalInput.value = '';
        S<HTMLSelectElement>('#sex')!.value = "";


        editingRecordId = null;
        handleClearPatientDocument();
        currentClinicalImagesData = [];
        renderClinicalImageGalleryUI();
        currentXRayImagesData = [];
        renderXRayImageGalleryUI();
        currentNewlyAddedLabReports = [];
        renderPendingLabReportsList();

        const otherMOIFieldGroup = S<HTMLDivElement>('#otherMOI-group');
        if (otherMOIFieldGroup) otherMOIFieldGroup.style.display = 'none';
        const otherMOIField = S<HTMLInputElement>('#otherMOI');
        if (otherMOIField) otherMOIField.value = '';
        const modeOfInjuryField = S<HTMLSelectElement>('#modeOfInjury');
        if (modeOfInjuryField) modeOfInjuryField.value = '';
        SAll<HTMLInputElement>('input[name="diagnosisSide"]').forEach(checkbox => checkbox.checked = false);
        showToast('Form cleared.', 'info');
    });
}

function renderPendingLabReportsList() {
    const container = S<HTMLDivElement>('#patient-form-lab-reports-pending-list');
    if (!container) return;

    if (currentNewlyAddedLabReports.length === 0) {
        container.innerHTML = '<p class="text-muted-color"><em>No lab reports staged for upload in this session.</em></p>';
        return;
    }

    container.innerHTML = `
        <p><strong>Lab Reports Staged for Upload:</strong></p>
        <ul>
            ${currentNewlyAddedLabReports.map(entry => `
                <li class="pending-lab-report-item" data-id="${entry.id}">
                    <i class="fas fa-${entry.fileType === 'application/pdf' ? 'file-pdf' : 'file-image'}"></i>
                    ${entry.fileName}
                    (Date: ${entry.reportDate || 'N/A'}, Params: ${entry.labParameters ? Object.keys(entry.labParameters).length : 0})
                    <button type="button" class="btn btn-danger-outline btn-sm remove-pending-lab-btn" data-id="${entry.id}" aria-label="Remove staged lab report">&times;</button>
                </li>
            `).join('')}
        </ul>
    `;
}


function renderClinicalImageGalleryUI() {
    const gallery = S<HTMLDivElement>('#clinical-images-gallery');
    const helperText = S<HTMLParagraphElement>('#clinical-images-helper-text');
    if (!gallery || !helperText) return;

    gallery.innerHTML = '';

    currentClinicalImagesData.forEach(img => {
        const card = document.createElement('div');
        card.className = 'gallery-image-card';
        card.innerHTML = `
            <img src="${img.dataUrl}" alt="${img.fileName || 'Clinical Image'}" title="${img.fileName || 'Clinical Image'}">
            <button type="button" class="delete-image-btn" data-id="${img.id}" aria-label="Remove image">&times;</button>
        `;
        gallery.appendChild(card);
    });

    if (currentClinicalImagesData.length < MAX_CLINICAL_IMAGES) {
        const addCard = document.createElement('div');
        addCard.className = 'gallery-add-card';
        addCard.setAttribute('role', 'button');
        addCard.setAttribute('tabindex', '0');
        addCard.setAttribute('aria-label', 'Add clinical image');
        addCard.innerHTML = '<i class="fas fa-plus"></i><span>Add Image</span>';
        gallery.appendChild(addCard);
    }
    helperText.textContent = `${currentClinicalImagesData.length}/${MAX_CLINICAL_IMAGES} images uploaded.`;
}

function renderXRayImageGalleryUI() {
    const gallery = S<HTMLDivElement>('#xray-images-gallery');
    const helperText = S<HTMLParagraphElement>('#xray-images-helper-text');
    if (!gallery || !helperText) return;

    gallery.innerHTML = '';

    currentXRayImagesData.forEach(img => {
        const card = document.createElement('div');
        card.className = 'gallery-image-card';
        card.innerHTML = `
            <img src="${img.dataUrl}" alt="${img.fileName || 'X-ray Image'}" title="${img.fileName || 'X-ray Image'}">
            <button type="button" class="delete-image-btn" data-id="${img.id}" aria-label="Remove X-ray image">&times;</button>
        `;
        gallery.appendChild(card);
    });

    if (currentXRayImagesData.length < MAX_XRAY_IMAGES) {
        const addCard = document.createElement('div');
        addCard.className = 'gallery-add-card';
        addCard.setAttribute('role', 'button');
        addCard.setAttribute('tabindex', '0');
        addCard.setAttribute('aria-label', 'Add X-ray image');
        addCard.innerHTML = '<i class="fas fa-plus"></i><span>Add X-ray</span>';
        gallery.appendChild(addCard);
    }
    helperText.textContent = `${currentXRayImagesData.length}/${MAX_XRAY_IMAGES} X-ray images uploaded.`;
}


async function handlePatientDetailsFormSubmit(event: Event, recordIdForEdit: string | null) {
    event.preventDefault();
    if (!derivedEncryptionKey && (currentUser?.role === 'user' || currentUser?.role === 'admin')) { // Admins always need key
        showToast("Warning: Encryption key not available. Data cannot be saved securely.", "warning");
        return;
    }

    const form = event.target as HTMLFormElement;
    const formData = new FormData(form);

    const patientIdValue = formData.get('patientIdInternal') as string;
    if (!/^KAHS\d+$/.test(patientIdValue)) {
        showToast('Patient ID must be in KAHS###### format (e.g., KAHS12345).', 'error');
        S<HTMLInputElement>('#patientIdInternal')?.focus();
        return;
    }

    const contactNumberValue = formData.get('contactNumber') as string;
    if (contactNumberValue && !/^\+977\d{9,10}$/.test(contactNumberValue)) {
        showToast('Contact number, if provided, must start with +977 followed by 9 or 10 digits.', 'error');
        S<HTMLInputElement>('#contactNumber')?.focus();
        return;
    }

    const ageYearsRaw = formData.get('ageYears') as string;
    const ageMonthsRaw = formData.get('ageMonths') as string;
    const ageDaysRaw = formData.get('ageDays') as string;

    const ageYears = ageYearsRaw ? parseInt(ageYearsRaw) : undefined;
    const ageMonths = ageMonthsRaw ? parseInt(ageMonthsRaw) : undefined;
    const ageDays = ageDaysRaw ? parseInt(ageDaysRaw) : undefined;

    if (ageYears === undefined || isNaN(ageYears) || ageYears < 0) {
        showToast('Age (Years) is required and must be a valid non-negative number.', 'error');
        S<HTMLInputElement>('#ageYears')?.focus();
        return;
    }
    if (ageMonths !== undefined && (isNaN(ageMonths) || ageMonths < 0 || ageMonths > 11)) {
        showToast('Age (Months) must be a valid number between 0 and 11, if provided.', 'error');
        S<HTMLInputElement>('#ageMonths')?.focus();
        return;
    }
    if (ageDays !== undefined && (isNaN(ageDays) || ageDays < 0 || ageDays > 31)) {
        showToast('Age (Days) must be a valid number between 0 and 31, if provided.', 'error');
        S<HTMLInputElement>('#ageDays')?.focus();
        return;
    }

    const ageData: Age = { years: ageYears };
    if (ageMonths !== undefined) ageData.months = ageMonths;
    if (ageDays !== undefined) ageData.days = ageDays;


    const recordData: Partial<Omit<TraumaRecord, 'age' | 'labReportFiles'>> & { age?: Age } = {
        patientId: patientIdValue,
        beemaNumber: extractedBeemaNumber || undefined,
        patientName: formData.get('patientName') as string,
        age: ageData,
        sex: formData.get('sex') as 'Male' | 'Female' | 'Other',
        address: formData.get('address') as string,
        contactNumber: contactNumberValue,

        patientDocumentUrl: patientDocumentBase64 || undefined,
        patientDocumentOcrText: formData.get('patientDocumentOcrText') as string || undefined,
        chiefComplaints: formData.get('chiefComplaints') as string || undefined,
        dateOfInjuryAD: formData.get('dateOfInjuryAD') as string,
        dateOfInjuryBS: formData.get('dateOfInjuryBS') as string,
        timeOfInjury: formData.get('timeOfInjury') as string,
        modeOfInjury: formData.get('modeOfInjury') as string,
        otherMOI: formData.get('modeOfInjury') === 'Other' ? formData.get('otherMOI') as string : undefined,
        siteOfInjury: formData.get('siteOfInjury') as string,
        typeOfInjury: formData.get('typeOfInjury') as string,
        descriptionOfInjuries: formData.get('descriptionOfInjuries') as string || undefined,
        presentComplaint: formData.get('presentComplaint') as string,
        glasgowComaScale: formData.get('glasgowComaScale') as string,
        vitalSigns: formData.get('vitalSigns') as string,
        systemicExamination: formData.get('systemicExamination') as string,
        localExamination: formData.get('localExamination') as string,
        provisionalDiagnosis: formData.get('provisionalDiagnosis') as string,
        diagnosisSide: Array.from(form.querySelectorAll<HTMLInputElement>('input[name="diagnosisSide"]:checked')).map(cb => cb.value as 'Right' | 'Left'),
        clinicalImageUrls: currentClinicalImagesData.map(img => img.dataUrl),
        xrayImageUrls: currentXRayImagesData.map(img => img.dataUrl),
    };

    if (!recordData.patientName || !recordData.age || recordData.age.years === undefined || !recordData.sex || !recordData.dateOfInjuryAD || !recordData.timeOfInjury || !recordData.modeOfInjury || !recordData.siteOfInjury || !recordData.typeOfInjury || !recordData.presentComplaint || !recordData.glasgowComaScale || !recordData.vitalSigns || !recordData.systemicExamination || !recordData.localExamination || !recordData.provisionalDiagnosis) {
        showToast('Please fill all required fields marked with *.', 'error');
        return;
    }

    if (recordIdForEdit) {
        const recordIndex = traumaRecords.findIndex(r => r.id === recordIdForEdit);
        if (recordIndex > -1) {
            const existingRecord = traumaRecords[recordIndex];
            const combinedLabReports = (existingRecord.labReportFiles || []).concat(currentNewlyAddedLabReports);

            traumaRecords[recordIndex] = {
                ...existingRecord,
                ...recordData,
                labReportFiles: combinedLabReports,
                updatedAt: new Date().toISOString(),
                updatedBy: currentUser!.id, // Track who updated
            } as TraumaRecord;
            showToast('Patient record updated successfully!', 'success');
        } else {
            showToast('Error: Record to update not found.', 'error');
            return;
        }
    } else {
        // Generate a system record ID (this is different from patientId)
        const recordSystemId = `REC-${formData.get('patientName') as string || 'Unknown'}-${Date.now()}`;

        const newRecord: TraumaRecord = {
            id: recordSystemId, // System record ID
            patientId: recordData.patientId!, // KAHS Patient ID
            beemaNumber: recordData.beemaNumber,
            patientName: recordData.patientName!,
            age: recordData.age!,
            sex: recordData.sex!,
            address: recordData.address!,
            contactNumber: recordData.contactNumber!,
            patientDocumentUrl: recordData.patientDocumentUrl,
            patientDocumentOcrText: recordData.patientDocumentOcrText,
            chiefComplaints: recordData.chiefComplaints,
            dateOfInjuryAD: recordData.dateOfInjuryAD!,
            dateOfInjuryBS: recordData.dateOfInjuryBS!,
            timeOfInjury: recordData.timeOfInjury!,
            modeOfInjury: recordData.modeOfInjury!,
            otherMOI: recordData.otherMOI,
            siteOfInjury: recordData.siteOfInjury!,
            typeOfInjury: recordData.typeOfInjury!,
            descriptionOfInjuries: recordData.descriptionOfInjuries,
            xrayImageUrls: recordData.xrayImageUrls,
            presentComplaint: recordData.presentComplaint!,
            glasgowComaScale: recordData.glasgowComaScale!,
            vitalSigns: recordData.vitalSigns!,
            systemicExamination: recordData.systemicExamination!,
            localExamination: recordData.localExamination!,
            diagnosisSide: recordData.diagnosisSide,
            provisionalDiagnosis: recordData.provisionalDiagnosis!,
            clinicalImageUrls: recordData.clinicalImageUrls,
            labReportFiles: currentNewlyAddedLabReports,
            createdBy: currentUser!.id,
            createdAt: new Date().toISOString(),
        };
        traumaRecords.push(newRecord);
        showToast('Patient record added successfully!', 'success');
    }

    await encryptAndSaveAllData();

    form.reset();
    S<HTMLSelectElement>('#sex')!.value = "";
    const dateOfInjuryBSInput = S<HTMLInputElement>('#dateOfInjuryBS');
    if (dateOfInjuryBSInput) dateOfInjuryBSInput.value = adbs.ad2bs(new Date().toISOString().split('T')[0]);

    const recordIdDisplayInput = S<HTMLInputElement>('#recordIdDisplay');
    if (recordIdDisplayInput) recordIdDisplayInput.value = 'New Record';
    const patientIdInternalInput = S<HTMLInputElement>('#patientIdInternal');
    if (patientIdInternalInput) patientIdInternalInput.value = '';
    const beemaNumberDisplayEl = S<HTMLInputElement>('#beemaNumberDisplay');
    if(beemaNumberDisplayEl) {
        beemaNumberDisplayEl.value = 'Not extracted';
        beemaNumberDisplayEl.style.display = 'block';
    }


    editingRecordId = null;
    patientDocumentFile = null;
    patientDocumentBase64 = null;
    extractedBeemaNumber = null;
    currentNewlyAddedLabReports = [];
    renderPendingLabReportsList();

    const patientDocPreviewArea = S<HTMLDivElement>('#patient-document-preview-area');
    if (patientDocPreviewArea) patientDocPreviewArea.innerHTML = 'No document uploaded/captured yet.';
    const clearPatientDocBtn = S<HTMLButtonElement>('#clearPatientDocumentBtn');
    if (clearPatientDocBtn) clearPatientDocBtn.remove();
    const patientDocOcrTextArea = S<HTMLTextAreaElement>('#patientDocumentOcrText');
    if (patientDocOcrTextArea) patientDocOcrTextArea.value = '';
    currentClinicalImagesData = [];
    renderClinicalImageGalleryUI();
    currentXRayImagesData = [];
    renderXRayImageGalleryUI();
    const otherMOIGroup = S<HTMLDivElement>('#otherMOI-group');
    if (otherMOIGroup) otherMOIGroup.style.display = 'none';
    const otherMOIInput = S<HTMLInputElement>('#otherMOI');
    if (otherMOIInput) otherMOIInput.value = '';
    const modeOfInjurySelect = S<HTMLSelectElement>('#modeOfInjury');
    if (modeOfInjurySelect) modeOfInjurySelect.value = '';
    SAll<HTMLInputElement>('input[name="diagnosisSide"]').forEach(cb => cb.checked = false);
}
// --- END Patient Details Form ---

// --- START Nurse Upload View Sections ---

function renderNursePatientDocUploadSection(): string {
    return `
        <div class="nurse-upload-section">
            <h3><i class="fas fa-id-card-alt"></i> Upload Patient Document</h3>
            <p class="instruction-text">Upload or capture an image of the patient's ID card, referral, or Beema card. Details will be auto-extracted by OCR. Files >5MB will be compressed (images only).</p>
            <div class="document-actions">
                <input type="file" id="nursePatientDocumentInput" accept="image/jpeg,image/png,image/gif,application/pdf" style="display:none;">
                <button type="button" id="nurseUploadPatientDocumentBtn" class="btn btn-primary btn-lg"><i class="fas fa-upload"></i> Upload Document File</button>
                <button type="button" id="nurseCapturePatientDocumentBtn" class="btn btn-secondary btn-lg"><i class="fas fa-camera"></i> Capture with Camera</button>
            </div>
            <div id="nurse-patient-doc-ocr-indicator" class="ocr-indicator" style="display: none;">
                <div class="spinner spinner-inline"></div>
                <span>Processing Patient Document OCR...</span>
            </div>
            <div id="nurse-patient-doc-preview-area" class="mt-2" style="max-height: 200px; overflow: auto; text-align:center;">
                 <p><em>Document preview will appear here.</em></p>
            </div>
            <div id="nurse-patient-doc-ocr-summary" class="mt-2" style="font-size: 0.95em; padding:10px; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9;">
                <p><em>OCR extracted details will appear here after upload.</em></p>
            </div>
        </div>
    `;
}

function renderNurseLabReportUploadSection(): string {
    const activePatient = currentActivePatientIdForNurse ? traumaRecords.find(r => r.patientId === currentActivePatientIdForNurse) : null;
    const activePatientInfo = activePatient
        ? `Currently selected for lab upload: <strong>${activePatient.patientName} (ID: ${activePatient.patientId})</strong>`
        : 'No patient selected. Please search and select a patient below, or process a new patient document.';

    return `
        <div class="nurse-upload-section">
            <h3><i class="fas fa-flask"></i> Upload Lab Report</h3>
            <p class="instruction-text">
                Search and select an existing patient OR process a new patient document in the "Upload Patient Document" tab first.
                The lab report will be linked to the selected patient. Max file size for images/PDFs is 10MB (images >5MB will be compressed).
            </p>
            <div id="nurse-active-patient-info" class="mb-2 instruction-text" style="background-color: #e6f7ff; border-color: #91d5ff;">${activePatientInfo}</div>

            <div class="search-bar mb-2">
                 <input type="text" id="nurseLabPatientSearchInput" placeholder="Search Existing Patient by Name or Patient ID (KAHS...)">
                 <button type="button" id="nurseLabPatientSearchBtn" class="btn btn-secondary btn-sm"><i class="fas fa-search"></i> Search Patients</button>
            </div>
            <div id="nurse-lab-patient-search-results" class="records-table-container" style="max-height: 200px; overflow-y: auto;">
                <p><em>Search results will appear here.</em></p>
            </div>

            <input type="file" id="nurseLabReportInput" accept=".pdf,.jpg,.jpeg,.png" style="display:none;" ${!currentActivePatientIdForNurse ? 'disabled' : ''}>
            <button type="button" id="nurseUploadLabReportBtn" class="btn btn-primary btn-lg mt-3" ${!currentActivePatientIdForNurse ? 'disabled' : ''}><i class="fas fa-upload"></i> Upload Lab Report for Selected Patient</button>
            <div id="nurse-lab-ocr-indicator" class="ocr-indicator" style="display: none;">
                <div class="spinner spinner-inline"></div>
                <span>Processing Lab Report OCR...</span>
            </div>
            <div id="nurse-lab-report-preview-area" class="mt-2" style="max-height: 200px; overflow: auto; text-align:center;">
                 <p><em>Lab report preview will appear here.</em></p>
            </div>
        </div>
    `;
}

function renderNurseViewLabHistorySection(): string {
    const activePatient = currentActivePatientIdForNurse ? traumaRecords.find(r => r.patientId === currentActivePatientIdForNurse) : null;
    let labHistoryHtml = '<p><em>No lab reports found for this patient or no patient selected.</em></p>';

    if (activePatient && activePatient.labReportFiles && activePatient.labReportFiles.length > 0) {
        const sortedReports = activePatient.labReportFiles.slice().sort((a,b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
        labHistoryHtml = sortedReports.map(report => `
            <div class="nurse-lab-report-card">
                <p><strong>File:</strong> ${escapeHtml(report.fileName)}</p>
                <p><strong>Date:</strong> ${escapeHtml(report.reportDate || 'N/A')}</p>
                <p><strong>Added On:</strong> ${new Date(report.createdAt).toLocaleDateString()}</p>
                ${report.rawOcrText ? `<p><strong>OCR Summary:</strong> <span class="ocr-summary-preview">${escapeHtml(report.rawOcrText.split('\n')[0])}...</span></p>` : ''}
                <button type="button" class="btn btn-info btn-sm nurse-view-lab-report-btn" data-lab-entry-id="${report.id}"><i class="fas fa-eye"></i> View Full Report</button>
            </div>
        `).join('');
    } else if (activePatient) {
        labHistoryHtml = '<p><em>No lab reports found for this patient.</em></p>';
    }

    return `
        <div class="nurse-upload-section">
            <h3><i class="fas fa-history"></i> View Previous Lab Reports</h3>
            <p class="instruction-text">Search and select a patient to view their lab report history. Reports are read-only.</p>
            
            <div class="search-bar mb-2">
                 <input type="text" id="nurseViewHistoryPatientSearchInput" placeholder="Search Patient by Name or ID (KAHS...)">
                 <button type="button" id="nurseViewHistoryPatientSearchBtn" class="btn btn-secondary btn-sm"><i class="fas fa-search"></i> Search Patients</button>
            </div>
             <div id="nurse-view-history-active-patient-info" class="mb-2 instruction-text" style="background-color: #e6f7ff; border-color: #91d5ff;">
                ${activePatient ? `Viewing reports for: <strong>${escapeHtml(activePatient.patientName)} (ID: ${escapeHtml(activePatient.patientId)})</strong>` : 'No patient selected.'}
            </div>
            <div id="nurse-view-history-patient-search-results" class="records-table-container" style="max-height: 200px; overflow-y: auto;">
                 <p><em>Search results for patient selection will appear here.</em></p>
            </div>

            <div id="nurse-lab-history-container" class="mt-3">
                ${labHistoryHtml}
            </div>
        </div>
    `;
}
// --- END Nurse Upload View Sections ---

// --- START Nurse Upload Listeners (Factored out) ---
function attachNursePatientDocUploadListeners() {
    const nursePatientDocumentInput = S<HTMLInputElement>('#nursePatientDocumentInput');
    const nurseUploadPatientDocumentBtn = S<HTMLButtonElement>('#nurseUploadPatientDocumentBtn');
    const nurseCapturePatientDocumentBtn = S<HTMLButtonElement>('#nurseCapturePatientDocumentBtn');
    const nursePatientDocPreviewArea = S<HTMLDivElement>('#nurse-patient-doc-preview-area');
    const nursePatientDocOcrSummary = S<HTMLDivElement>('#nurse-patient-doc-ocr-summary');

    nurseUploadPatientDocumentBtn?.addEventListener('click', () => nursePatientDocumentInput?.click());
    nurseCapturePatientDocumentBtn?.addEventListener('click', () => openCameraModal('patientDocument')); // Reusing context for simplicity, UI will handle file

    const handlePatientDocUpload = async (fileToProcess: File) => {
        if (!derivedEncryptionKey) {
            showToast("Warning: Admin key not set. Data cannot be saved securely.", "warning");
        }

        if (nursePatientDocPreviewArea) nursePatientDocPreviewArea.innerHTML = `<p class="text-info"><i class="fas fa-spinner fa-spin"></i> Processing ${escapeHtml(fileToProcess.name)}...</p>`;
        if (nursePatientDocOcrSummary) nursePatientDocOcrSummary.innerHTML = '';

        if (fileToProcess.size > LARGE_FILE_THRESHOLD_BYTES && fileToProcess.type.startsWith('image/')) {
            showToast("Patient document image is large, attempting compression...", "info");
            try {
                const compressedBase64 = await compressImage(fileToProcess, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_QUALITY);
                fileToProcess = await base64StringToFile(compressedBase64, fileToProcess.name, fileToProcess.type);
                showToast("Compression successful.", "success");
            } catch (e) { console.error("Compression error:", e); showToast("Compression failed. Using original.", "warning"); }
        }

        const docBase64 = await fileToDataURL(fileToProcess);

        if (nursePatientDocPreviewArea) {
             if (fileToProcess.type === 'application/pdf') {
                nursePatientDocPreviewArea.innerHTML = `<p><i class="fas fa-file-pdf text-danger" style="font-size: 2.5rem;"></i> PDF Uploaded: ${escapeHtml(fileToProcess.name)}</p>`;
            } else {
                nursePatientDocPreviewArea.innerHTML = `<img src="${docBase64}" alt="Patient Doc Preview"> <p>${escapeHtml(fileToProcess.name)}</p>`;
            }
        }

        const ocrResult = await performOcr(fileToProcess, 'patient_document');

        if (ocrResult.error && !ocrResult.extractedDemographics && !ocrResult.rawText) {
            showToast(`OCR Failed: ${ocrResult.error || 'No data extracted.'}`, "error");
            if (nursePatientDocOcrSummary) nursePatientDocOcrSummary.innerHTML = `<p class="text-danger">OCR Failed: ${ocrResult.error || 'No data extracted.'}</p>`;
            return;
        }

        const demographics = ocrResult.extractedDemographics || {};
        let finalPatientId = demographics.patientId;

        if (!finalPatientId || !/^KAHS\d+$/.test(finalPatientId)) {
            finalPatientId = generateNewUniqueKahsPatientId();
            showToast(`New Patient ID generated: ${finalPatientId}`, "info");
        }

        let record = traumaRecords.find(r => r.patientId === finalPatientId);
        const now = new Date().toISOString();
        const minimalAge: Age = demographics.age ? { years: demographics.age.years, months: demographics.age.months, days: demographics.age.days }
                               : (demographics.ageYears !== undefined ? { years: demographics.ageYears, months: demographics.ageMonths, days: demographics.ageDays } : { years: 0 });

        if (record) {
            record.patientName = demographics.patientName || record.patientName || 'N/A';
            record.age = minimalAge.years > 0 || (record.age.years || 0) > 0 || Object.keys(minimalAge).length > 1 ? minimalAge : record.age;
            record.sex = demographics.sex || record.sex || 'Other';
            record.address = demographics.address || record.address || 'N/A';
            record.contactNumber = demographics.contactNumber || record.contactNumber || 'N/A';
            record.beemaNumber = demographics.beemaNumber || record.beemaNumber;
            record.patientDocumentUrl = docBase64;
            record.patientDocumentOcrText = ocrResult.rawText || record.patientDocumentOcrText;
            record.updatedAt = now;
            record.updatedBy = currentUser!.id;
            showToast(`Patient record for ID ${finalPatientId} updated successfully.`, "success");
        } else {
            const newRecordSystemId = `NREC-${(demographics.patientName || 'Unknown').replace(/\s+/g, '')}-${Date.now()}`;
            record = {
                id: newRecordSystemId, patientId: finalPatientId,
                patientName: demographics.patientName || 'Unknown (OCR)', age: minimalAge, sex: demographics.sex || 'Other',
                address: demographics.address || 'Unknown (OCR)', contactNumber: demographics.contactNumber || 'Unknown (OCR)',
                beemaNumber: demographics.beemaNumber, patientDocumentUrl: docBase64, patientDocumentOcrText: ocrResult.rawText,
                createdBy: currentUser!.id, createdAt: now,
                dateOfInjuryAD: '', timeOfInjury: '', modeOfInjury: '', siteOfInjury: '', typeOfInjury: '',
                presentComplaint: '', glasgowComaScale: '', vitalSigns: '', systemicExamination: '', localExamination: '', provisionalDiagnosis: '', dateOfInjuryBS: '',
            };
            traumaRecords.push(record);
            showToast(`New patient record created with ID ${finalPatientId} successfully.`, "success");
        }

        if (nursePatientDocOcrSummary) {
             let summaryHtml = `<p class="text-success">✅ Patient data captured successfully for ID: <strong>${finalPatientId}</strong></p><ul>`;
             if(record.patientName && record.patientName !== 'Unknown (OCR)') summaryHtml += `<li>Name: ${escapeHtml(record.patientName)}</li>`;
             if(record.age.years > 0 || (record.age.months || 0) > 0 || (record.age.days || 0) > 0) summaryHtml += `<li>Age: ${formatAge(record.age)}</li>`;
             if(record.sex !== 'Other' || demographics.sex) summaryHtml += `<li>Sex: ${escapeHtml(record.sex)}</li>`;
             if(record.beemaNumber) summaryHtml += `<li>Beema No: ${escapeHtml(record.beemaNumber)}</li>`;
             if(record.address && record.address !== 'Unknown (OCR)') summaryHtml += `<li>Address: ${escapeHtml(record.address)}</li>`;
             if(record.contactNumber && record.contactNumber !== 'Unknown (OCR)') summaryHtml += `<li>Contact: ${escapeHtml(record.contactNumber)}</li>`;
             summaryHtml += `</ul>`;
             nursePatientDocOcrSummary.innerHTML = summaryHtml;
        }

        currentActivePatientIdForNurse = finalPatientId; // Set active patient for other tabs
        // Optionally, update other sections if they are already rendered and depend on this ID
        if(S<HTMLDivElement>('#nurse-active-patient-info')) S<HTMLDivElement>('#nurse-active-patient-info')!.innerHTML = `Currently selected for lab upload: <strong>${escapeHtml(record.patientName)} (ID: ${escapeHtml(finalPatientId)})</strong>`;
        if(S<HTMLButtonElement>('#nurseUploadLabReportBtn')) S<HTMLButtonElement>('#nurseUploadLabReportBtn')!.disabled = false;
        if(S<HTMLInputElement>('#nurseLabReportInput')) S<HTMLInputElement>('#nurseLabReportInput')!.disabled = false;


        if (derivedEncryptionKey) await encryptAndSaveAllData();
        else showToast("Data processed locally. Admin login required to save permanently.", "warning");
        if (nursePatientDocumentInput) nursePatientDocumentInput.value = '';
    };


    nursePatientDocumentInput?.addEventListener('change', async (event) => {
        const files = (event.target as HTMLInputElement).files;
        if (files && files[0]) await handlePatientDocUpload(files[0]);
    });

    // Listener for camera capture (if we re-add patientDocument context for camera in nurse view)
    // This part is tricky if openCameraModal directly tries to update patient-details-form elements.
    // For now, nurse view uses its own dedicated inputs. If openCameraModal is generic enough,
    // we would need a callback here to process the capturedFile.
    // For now, the capture button will use patientDocument context, and saveBtn in camera modal will call handlePatientDocUpload.
}


function attachNurseLabReportUploadListeners() {
    const nurseLabReportInput = S<HTMLInputElement>('#nurseLabReportInput');
    const nurseUploadLabReportBtn = S<HTMLButtonElement>('#nurseUploadLabReportBtn');
    const nurseLabReportPreviewArea = S<HTMLDivElement>('#nurse-lab-report-preview-area');
    const nurseActivePatientInfo = S<HTMLDivElement>('#nurse-active-patient-info');
    const nurseLabPatientSearchInput = S<HTMLInputElement>('#nurseLabPatientSearchInput');
    const nurseLabPatientSearchBtn = S<HTMLButtonElement>('#nurseLabPatientSearchBtn');
    const nurseLabPatientSearchResultsDiv = S<HTMLDivElement>('#nurse-lab-patient-search-results');

    const updateActivePatientDisplay = () => {
        const patient = currentActivePatientIdForNurse ? traumaRecords.find(r => r.patientId === currentActivePatientIdForNurse) : null;
        if (nurseActivePatientInfo) {
            nurseActivePatientInfo.innerHTML = patient
                ? `Currently selected for lab upload: <strong>${escapeHtml(patient.patientName)} (ID: ${escapeHtml(patient.patientId)})</strong>`
                : 'No patient selected. Please search and select a patient below, or process a new patient document.';
        }
        if (nurseUploadLabReportBtn) nurseUploadLabReportBtn.disabled = !patient;
        if (nurseLabReportInput) nurseLabReportInput.disabled = !patient;
    };
    updateActivePatientDisplay(); // Initial state


    nurseLabPatientSearchBtn?.addEventListener('click', () => {
        const searchTerm = nurseLabPatientSearchInput?.value.toLowerCase().trim() || '';
        if (!searchTerm) {
            nurseLabPatientSearchResults = [];
            if (nurseLabPatientSearchResultsDiv) nurseLabPatientSearchResultsDiv.innerHTML = '<p class="text-muted-color"><em>Enter search term (Name or Patient ID).</em></p>';
            return;
        }
        nurseLabPatientSearchResults = traumaRecords.filter(r =>
            r.patientName.toLowerCase().includes(searchTerm) ||
            r.patientId.toLowerCase().includes(searchTerm)
        );
        renderNurseLabPatientSearchResults(nurseLabPatientSearchResultsDiv, 'labReportUpload');
    });

    nurseLabPatientSearchResultsDiv?.addEventListener('click', (event) => {
        const target = event.target as HTMLElement;
        const selectBtn = target.closest<HTMLButtonElement>('.nurse-select-patient-for-lab-btn');
        if (selectBtn) {
            const patientId = selectBtn.dataset.patientId;
            const patient = traumaRecords.find(r => r.patientId === patientId);
            if (patient) {
                currentActivePatientIdForNurse = patient.patientId;
                updateActivePatientDisplay();
                if (nurseLabPatientSearchResultsDiv) nurseLabPatientSearchResultsDiv.innerHTML = `<p class="text-success p-2">Selected: ${escapeHtml(patient.patientName)} (${escapeHtml(patient.patientId)})</p>`;
                showToast(`Patient ${escapeHtml(patient.patientId)} selected for lab report upload.`, "info");
            }
        }
    });

    nurseUploadLabReportBtn?.addEventListener('click', () => {
        if (!currentActivePatientIdForNurse) {
            showToast("Please select an existing patient to link the lab report.", "warning");
            return;
        }
        nurseLabReportInput?.click();
    });

    nurseLabReportInput?.addEventListener('change', async (event) => {
        if (!currentActivePatientIdForNurse) {
            showToast("Error: No active patient ID for lab report linking.", "error"); return;
        }
        const files = (event.target as HTMLInputElement).files;
        if (!files || !files[0]) return;
        let fileToProcess = files[0];

        if (!derivedEncryptionKey) { showToast("Warning: Admin key not set. Lab report cannot be saved securely.", "warning"); }
        if (nurseLabReportPreviewArea) nurseLabReportPreviewArea.innerHTML = `<p class="text-info"><i class="fas fa-spinner fa-spin"></i> Processing ${escapeHtml(fileToProcess.name)}...</p>`;

        if (fileToProcess.size > LARGE_FILE_THRESHOLD_BYTES && fileToProcess.type.startsWith('image/')) {
             showToast("Lab report image is large, compressing...", "info");
            try {
                const compressedBase64 = await compressImage(fileToProcess, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_QUALITY);
                fileToProcess = await base64StringToFile(compressedBase64, fileToProcess.name, fileToProcess.type);
                 showToast("Compression successful.", "success");
            } catch (e) { console.error("Compression error:", e); showToast("Compression failed. Using original.", "warning");}
        }

        const labBase64 = await fileToDataURL(fileToProcess);

        if (nurseLabReportPreviewArea) {
             if (fileToProcess.type === 'application/pdf') {
                nurseLabReportPreviewArea.innerHTML = `<p><i class="fas fa-file-pdf text-danger" style="font-size: 2.5rem;"></i> PDF Lab Report Uploaded: ${escapeHtml(fileToProcess.name)}</p>`;
            } else {
                nurseLabReportPreviewArea.innerHTML = `<img src="${labBase64}" alt="Lab Report Preview"> <p>${escapeHtml(fileToProcess.name)}</p>`;
            }
        }

        const ocrResult = await performOcr(fileToProcess, 'lab_report');
        const record = traumaRecords.find(r => r.patientId === currentActivePatientIdForNurse);

        if (!record) { showToast(`Error: Patient record ${currentActivePatientIdForNurse} not found.`, "error"); return; }

        if (ocrResult.error && !ocrResult.extractedLabData && !ocrResult.rawText) {
            showToast(`Lab OCR Failed: ${ocrResult.error || 'No data extracted.'}`, "error");
            const errorLabEntry: LabReportEntry = { id: uuid(), fileName: fileToProcess.name, fileUrl: labBase64, fileType: fileToProcess.type, rawOcrText: ocrResult.rawText || `OCR Error: ${ocrResult.error || 'No data extracted.'}`, source: 'auto-captured', createdAt: new Date().toISOString() };
            if (!record.labReportFiles) record.labReportFiles = [];
            record.labReportFiles.push(errorLabEntry);
        } else {
            const newLabEntry: LabReportEntry = { id: uuid(), fileName: fileToProcess.name, fileUrl: labBase64, fileType: fileToProcess.type, reportDate: ocrResult.extractedLabData?.reportDate, labParameters: ocrResult.extractedLabData?.labParameters, rawOcrText: ocrResult.rawText, source: 'auto-captured', createdAt: new Date().toISOString() };
            if (!record.labReportFiles) record.labReportFiles = [];
            record.labReportFiles.push(newLabEntry);
            showToast(`✅ Lab Report Uploaded Successfully for Patient ID: ${currentActivePatientIdForNurse}`, "success");
            if (nurseLabReportPreviewArea) nurseLabReportPreviewArea.innerHTML += `<br><span class="text-success">Linked to ${currentActivePatientIdForNurse}.</span>`;
        }
        record.updatedAt = new Date().toISOString(); record.updatedBy = currentUser!.id;
        if (derivedEncryptionKey) await encryptAndSaveAllData();
        else showToast("Lab report processed locally. Admin login required to save permanently.", "warning");
        if (nurseLabReportInput) nurseLabReportInput.value = '';
    });
}


function attachNurseViewLabHistoryListeners() {
    const searchInput = S<HTMLInputElement>('#nurseViewHistoryPatientSearchInput');
    const searchBtn = S<HTMLButtonElement>('#nurseViewHistoryPatientSearchBtn');
    const searchResultsDiv = S<HTMLDivElement>('#nurse-view-history-patient-search-results');
    const activePatientInfoDiv = S<HTMLDivElement>('#nurse-view-history-active-patient-info');
    const labHistoryContainer = S<HTMLDivElement>('#nurse-lab-history-container');

    const updateLabHistoryDisplay = () => {
        const patient = currentActivePatientIdForNurse ? traumaRecords.find(r => r.patientId === currentActivePatientIdForNurse) : null;
        if(activePatientInfoDiv) {
            activePatientInfoDiv.innerHTML = patient
                ? `Viewing reports for: <strong>${escapeHtml(patient.patientName)} (ID: ${escapeHtml(patient.patientId)})</strong>`
                : 'No patient selected.';
        }

        if (patient && patient.labReportFiles && patient.labReportFiles.length > 0) {
            const sortedReports = patient.labReportFiles.slice().sort((a,b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
            labHistoryContainer!.innerHTML = sortedReports.map(report => `
                <div class="nurse-lab-report-card">
                    <p><strong>File:</strong> ${escapeHtml(report.fileName)}</p>
                    <p><strong>Date:</strong> ${escapeHtml(report.reportDate || 'N/A')}</p>
                    <p><strong>Added On:</strong> ${new Date(report.createdAt).toLocaleDateString()}</p>
                    ${report.rawOcrText ? `<p><strong>OCR Summary:</strong> <span class="ocr-summary-preview">${escapeHtml(report.rawOcrText.split('\n')[0])}...</span></p>` : ''}
                    <button type="button" class="btn btn-info btn-sm nurse-view-lab-report-btn" data-lab-entry-id="${report.id}"><i class="fas fa-eye"></i> View Full Report</button>
                </div>
            `).join('');
        } else if (patient) {
            labHistoryContainer!.innerHTML = '<p class="text-muted-color"><em>No lab reports found for this patient.</em></p>';
        } else {
            labHistoryContainer!.innerHTML = '<p class="text-muted-color"><em>Select a patient to view their lab report history.</em></p>';
        }
    };

    updateLabHistoryDisplay(); // Initial render based on currentActivePatientIdForNurse

    searchBtn?.addEventListener('click', () => {
        const searchTerm = searchInput?.value.toLowerCase().trim() || '';
        if (!searchTerm) {
            if (searchResultsDiv) searchResultsDiv.innerHTML = '<p class="text-muted-color"><em>Enter search term (Name or Patient ID).</em></p>';
            return;
        }
        const results = traumaRecords.filter(r =>
            r.patientName.toLowerCase().includes(searchTerm) ||
            r.patientId.toLowerCase().includes(searchTerm)
        );
        renderNurseLabPatientSearchResults(searchResultsDiv, 'viewLabHistory');
    });

    searchResultsDiv?.addEventListener('click', (event) => {
        const target = event.target as HTMLElement;
        const selectBtn = target.closest<HTMLButtonElement>('.nurse-select-patient-for-lab-btn'); // Reusing class for consistency
        if (selectBtn) {
            const patientId = selectBtn.dataset.patientId;
            const patient = traumaRecords.find(r => r.patientId === patientId);
            if (patient) {
                currentActivePatientIdForNurse = patient.patientId;
                updateLabHistoryDisplay();
                if (searchResultsDiv) searchResultsDiv.innerHTML = `<p class="text-success p-2">Selected: ${escapeHtml(patient.patientName)} (${escapeHtml(patient.patientId)})</p>`;
                showToast(`Viewing lab reports for ${escapeHtml(patient.patientId)}.`, "info");
            }
        }
    });

    labHistoryContainer?.addEventListener('click', (event) => {
        const target = event.target as HTMLElement;
        const viewBtn = target.closest<HTMLButtonElement>('.nurse-view-lab-report-btn');
        if (viewBtn) {
            const labEntryId = viewBtn.dataset.labEntryId;
            const patient = currentActivePatientIdForNurse ? traumaRecords.find(r => r.patientId === currentActivePatientIdForNurse) : null;
            const labEntry = patient?.labReportFiles?.find(entry => entry.id === labEntryId);
            if (labEntry) {
                let modalContentHtml = `<h4>Lab Report: ${escapeHtml(labEntry.fileName)}</h4>`;
                if (labEntry.fileUrl.startsWith('data:application/pdf')) {
                    modalContentHtml += `<p>This is a PDF document. <a href="${labEntry.fileUrl}" target="_blank" class="btn btn-primary btn-sm">Open PDF in New Tab</a></p>
                                         <p class="text-muted-color"><em>PDF preview in modal is not supported, please open in new tab.</em></p>`;
                } else if (labEntry.fileUrl.startsWith('data:image')) {
                    modalContentHtml += `<img src="${labEntry.fileUrl}" alt="Lab Report: ${escapeHtml(labEntry.fileName)}" style="max-width:100%; height:auto; border:1px solid #ccc;">`;
                } else {
                    modalContentHtml += `<p>Cannot preview this file type directly. <a href="${labEntry.fileUrl}" target="_blank">Open File</a></p>`;
                }
                if (labEntry.rawOcrText) {
                    modalContentHtml += `<h5>Extracted OCR Text:</h5><pre style="white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow-y: auto; background-color: #f8f9fa; padding:10px; border-radius:4px;">${escapeHtml(labEntry.rawOcrText)}</pre>`;
                }
                openModal(`View Lab Report - ${escapeHtml(labEntry.fileName)}`, modalContentHtml, 'lg', false);
            }
        }
    });
}

function renderNurseLabPatientSearchResults(container: HTMLDivElement | null, context: 'labReportUpload' | 'viewLabHistory') {
    if (!container) return;
    if (nurseLabPatientSearchResults.length === 0) {
        container.innerHTML = '<p class="text-muted-color"><em>No matching patients found.</em></p>';
        return;
    }
    // Using same class '.nurse-select-patient-for-lab-btn' for simplicity, context handled by specific listeners
    container.innerHTML = `
        <table class="records-table records-table-sm">
            <thead><tr><th>Patient ID</th><th>Name</th><th>Action</th></tr></thead>
            <tbody>
                ${nurseLabPatientSearchResults.map(r => `
                    <tr>
                        <td>${escapeHtml(r.patientId)}</td>
                        <td>${escapeHtml(r.patientName)}</td>
                        <td><button class="btn btn-sm btn-success nurse-select-patient-for-lab-btn" data-patient-id="${r.patientId}" data-context="${context}">
                            <i class="fas fa-check-circle"></i> Select
                        </button></td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

// --- END Nurse Upload Listeners ---


// --- START Camera Modal Logic ---
let cameraStream: MediaStream | null = null;
let currentCaptureContext: 'patientDocument' | 'clinicalImage' | null = null;
let capturedImageDataUrl: string | null = null;

async function openCameraModal(context: 'patientDocument' | 'clinicalImage') {
    currentCaptureContext = context;
    const modal = S<HTMLDivElement>('#camera-modal');
    const video = S<HTMLVideoElement>('#camera-feed');
    const captureBtn = S<HTMLButtonElement>('#capture-photo-btn');
    const saveBtn = S<HTMLButtonElement>('#save-captured-photo-btn');
    const retakeBtn = S<HTMLButtonElement>('#retake-photo-btn');
    const previewImg = S<HTMLImageElement>('#captured-photo-preview-modal');
    const modalTitle = S<HTMLHeadingElement>('#camera-modal-title');

    if (!modal || !video || !captureBtn || !saveBtn || !retakeBtn || !previewImg || !modalTitle) return;

    modalTitle.textContent = context === 'patientDocument' ? 'Capture Patient Document' : 'Capture Clinical Image';

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        video.srcObject = cameraStream;
        video.style.display = 'block';
        previewImg.style.display = 'none';
        captureBtn.style.display = 'inline-block';
        saveBtn.style.display = 'none';
        retakeBtn.style.display = 'none';
        modal.style.display = 'flex';
    } catch (err) {
        console.error("Error accessing camera:", err);
        showToast("Could not access camera. Check permissions.", "error");
    }
}

function closeCameraModal() {
    const modal = S<HTMLDivElement>('#camera-modal');
    if (modal) modal.style.display = 'none';
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    capturedImageDataUrl = null;
    currentCaptureContext = null;
}

function setupCameraModalListeners() {
    const video = S<HTMLVideoElement>('#camera-feed');
    const canvas = S<HTMLCanvasElement>('#photo-canvas');
    const captureBtn = S<HTMLButtonElement>('#capture-photo-btn');
    const saveBtn = S<HTMLButtonElement>('#save-captured-photo-btn');
    const retakeBtn = S<HTMLButtonElement>('#retake-photo-btn');
    const closeBtn = S<HTMLButtonElement>('#camera-modal-close-button');
    const previewImg = S<HTMLImageElement>('#captured-photo-preview-modal');

    closeBtn?.addEventListener('click', closeCameraModal);

    captureBtn?.addEventListener('click', () => {
        if (!video || !canvas || !previewImg || !captureBtn || !saveBtn || !retakeBtn) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

        capturedImageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
        previewImg.src = capturedImageDataUrl;

        video.style.display = 'none';
        previewImg.style.display = 'block';
        captureBtn.style.display = 'none';
        saveBtn.style.display = 'inline-block';
        retakeBtn.style.display = 'inline-block';
    });

    retakeBtn?.addEventListener('click', () => {
        if (!video || !previewImg || !captureBtn || !saveBtn || !retakeBtn) return;
        video.style.display = 'block';
        previewImg.style.display = 'none';
        previewImg.src = '#';
        captureBtn.style.display = 'inline-block';
        saveBtn.style.display = 'none';
        retakeBtn.style.display = 'none';
        capturedImageDataUrl = null;
    });

    saveBtn?.addEventListener('click', async () => {
        if (!capturedImageDataUrl || !currentCaptureContext) return;

        const fileName = `${currentCaptureContext}_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
        let capturedFile = await base64StringToFile(capturedImageDataUrl, fileName, 'image/jpeg');

        if (currentCaptureContext === 'patientDocument') {
            if (currentUser?.role === 'nurse' && currentNurseView === 'patientDocUpload') {
                // Nurse view patient document capture
                const nursePatientDocUploadHandler = S<HTMLInputElement>('#nursePatientDocumentInput');
                if (nursePatientDocUploadHandler) { // Use the patientDocUpload handler
                     // Manually construct and dispatch a change event for the file input
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(capturedFile);
                    nursePatientDocUploadHandler.files = dataTransfer.files;
                    nursePatientDocUploadHandler.dispatchEvent(new Event('change', { bubbles: true }));
                }

            } else if (S<HTMLFormElement>('#patient-details-form')) { // Main patient details form
                if (capturedFile.size > LARGE_FILE_THRESHOLD_BYTES) {
                    showToast("Captured document is large, attempting compression...", "info");
                    try {
                        const compressedBase64 = await compressImage(capturedFile, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_QUALITY);
                        capturedFile = await base64StringToFile(compressedBase64, capturedFile.name, 'image/jpeg');
                        patientDocumentBase64 = compressedBase64;
                        showToast("Compression successful.", "success");
                    } catch (e) { console.error("Compression error for captured doc:", e); showToast("Compression failed.", "error"); patientDocumentBase64 = capturedImageDataUrl;  }
                } else {
                     patientDocumentBase64 = capturedImageDataUrl;
                }
                patientDocumentFile = capturedFile;

                const patientDocPreviewArea = S<HTMLDivElement>('#patient-document-preview-area');
                if (patientDocPreviewArea) patientDocPreviewArea.innerHTML = `<img src="${patientDocumentBase64}" alt="Captured Patient Document">`;

                let clearBtn = S<HTMLButtonElement>('#clearPatientDocumentBtn');
                if (!clearBtn && patientDocPreviewArea) {
                     clearBtn = document.createElement('button');
                     clearBtn.type = 'button'; clearBtn.id = 'clearPatientDocumentBtn';
                     clearBtn.className = 'btn btn-danger-outline btn-sm mt-2';
                     clearBtn.innerHTML = '<i class="fas fa-trash-alt"></i> Clear Scanned Document';
                     clearBtn.addEventListener('click', () => {
                        patientDocumentFile = null; patientDocumentBase64 = null; extractedBeemaNumber = null;
                        S<HTMLInputElement>('#patientDocumentInput')?.value && (S<HTMLInputElement>('#patientDocumentInput')!.value = '');
                        S<HTMLDivElement>('#patient-document-preview-area') && (S<HTMLDivElement>('#patient-document-preview-area')!.innerHTML = 'No document uploaded/captured yet.');
                        S<HTMLTextAreaElement>('#patientDocumentOcrText') && (S<HTMLTextAreaElement>('#patientDocumentOcrText')!.value = '');
                        const beemaDisplay = S<HTMLInputElement>('#beemaNumberDisplay');
                        if(beemaDisplay) {
                            beemaDisplay.value = 'Not extracted';
                            beemaDisplay.style.display = 'block';
                        }
                        const recordIdDisp = S<HTMLInputElement>('#recordIdDisplay');
                         if (recordIdDisp) {
                            const currentPatientIdInput = S<HTMLInputElement>('#patientIdInternal');
                            let dispVal = `Patient ID: ${currentPatientIdInput?.value || (editingRecordId ? traumaRecords.find(r=>r.id===editingRecordId)?.patientId : '[Pending]')}`;
                            if (editingRecordId) { dispVal += ` (Record ID: ${editingRecordId})`; }
                            else if (!editingRecordId && !(currentPatientIdInput?.value || '').startsWith("KAHS")) { dispVal = "New Record"; }
                            recordIdDisp.value = dispVal;
                        }
                        clearBtn?.remove();
                     });
                     patientDocPreviewArea.insertAdjacentElement('afterend', clearBtn);
                }
                if(clearBtn) clearBtn.style.display = 'block';


                const ocrResult = await performOcr(capturedFile, 'patient_document');
                const ocrTextArea = S<HTMLTextAreaElement>('#patientDocumentOcrText');
                const beemaDisplay = S<HTMLInputElement>('#beemaNumberDisplay');
                const recordIdDisplayEl = S<HTMLInputElement>('#recordIdDisplay');
                const patientIdInput = S<HTMLInputElement>('#patientIdInternal');


                if (ocrResult.rawText && ocrTextArea) {
                    ocrTextArea.value = ocrResult.rawText;
                    if (!ocrResult.error) showToast("OCR processing complete.", "success");
                    else showToast(`OCR Information: ${ocrResult.error}`, "info");

                    const demographics = ocrResult.extractedDemographics;
                    if (demographics) {
                        if (demographics.patientName && S<HTMLInputElement>('#patientName')) {
                            (S<HTMLInputElement>('#patientName')!).value = demographics.patientName;
                        }
                        if (demographics.patientId && patientIdInput) {
                            patientIdInput.value = demographics.patientId;
                        }
                        const ageYearsEl = S<HTMLInputElement>('#ageYears');
                        const ageMonthsEl = S<HTMLInputElement>('#ageMonths');
                        const ageDaysEl = S<HTMLInputElement>('#ageDays');
                        if (demographics.age && typeof demographics.age.years === 'number') {
                            if (ageYearsEl) ageYearsEl.value = String(demographics.age.years);
                            if (ageMonthsEl) ageMonthsEl.value = demographics.age.months !== undefined ? String(demographics.age.months) : '';
                            if (ageDaysEl) ageDaysEl.value = demographics.age.days !== undefined ? String(demographics.age.days) : '';
                        } else if (demographics.ageYears !== undefined) {
                            if (ageYearsEl) ageYearsEl.value = String(demographics.ageYears);
                            if (ageMonthsEl) ageMonthsEl.value = demographics.ageMonths !== undefined ? String(demographics.ageMonths) : '';
                            if (ageDaysEl) ageDaysEl.value = demographics.ageDays !== undefined ? String(demographics.ageDays) : '';
                        }
                        if (demographics.sex && S<HTMLSelectElement>('#sex')) {
                            (S<HTMLSelectElement>('#sex')!).value = demographics.sex;
                        }
                        if (demographics.address && S<HTMLInputElement>('#address')) {
                            (S<HTMLInputElement>('#address')!).value = demographics.address;
                        }
                        if (demographics.contactNumber && S<HTMLInputElement>('#contactNumber')) {
                            (S<HTMLInputElement>('#contactNumber')!).value = demographics.contactNumber;
                        }
                        extractedBeemaNumber = demographics.beemaNumber || null;
                        if(beemaDisplay) {
                            if(extractedBeemaNumber) { beemaDisplay.value = extractedBeemaNumber; beemaDisplay.style.display = 'block'; }
                            else {
                                beemaDisplay.value = 'Not extracted';
                                beemaDisplay.style.display = 'block';
                            }
                        }
                        if(recordIdDisplayEl) {
                            let displayValue = `Patient ID: ${demographics.patientId || patientIdInput?.value || '[Pending]'}`;
                            if(extractedBeemaNumber) displayValue += ` | Beema No.: ${extractedBeemaNumber}`;
                             if (editingRecordId) displayValue += ` (Record ID: ${editingRecordId})`; else if (!editingRecordId && !(demographics.patientId || patientIdInput?.value || '').startsWith("KAHS")) displayValue = "New Record" + (extractedBeemaNumber ? ` | Beema No.: ${extractedBeemaNumber}` : "");
                            recordIdDisplayEl.value = displayValue;
                        }
                         if (Object.keys(demographics).length > 0) {
                            showToast("Demographics partially filled from OCR.", "info");
                        }
                    }
                } else if (ocrResult.error && ocrTextArea) {
                    ocrTextArea.value = `OCR Failed: ${ocrResult.error}`;
                    showToast(`OCR Error: ${ocrResult.error}`, "error");
                    extractedBeemaNumber = null;
                    if(beemaDisplay) {
                        beemaDisplay.value = 'Not extracted';
                        beemaDisplay.style.display = 'block';
                    }
                    if(recordIdDisplayEl && patientIdInput) {
                         let displayValue = `Patient ID: ${patientIdInput.value || '[Pending]'}`;
                         if (editingRecordId) displayValue += ` (Record ID: ${editingRecordId})`; else if (!editingRecordId && !(patientIdInput.value || '').startsWith("KAHS")) displayValue = "New Record";
                         recordIdDisplayEl.value = displayValue;
                    }
                }
            }
        } else if (currentCaptureContext === 'clinicalImage') {
            if (S<HTMLFormElement>('#patient-details-form')) {
                if (currentClinicalImagesData.length < MAX_CLINICAL_IMAGES) {
                     try {
                        let fileToCompress = capturedFile;
                        if (capturedFile.size > LARGE_FILE_THRESHOLD_BYTES) {
                             showToast(`Captured clinical image is large, attempting initial compression...`, "info");
                             const initialBase64 = await compressImage(capturedFile, LARGE_FILE_GENERAL_COMPRESSION_MAX_DIM, LARGE_FILE_GENERAL_COMPRESSION_MAX_DIM, LARGE_FILE_GENERAL_COMPRESSION_QUALITY);
                             fileToCompress = await base64StringToFile(initialBase64, capturedFile.name, 'image/jpeg');
                             showToast(`Initial compression for captured image successful.`, "success");
                        }
                        const compressedDataUrl = await compressImage(fileToCompress, CLINICAL_IMAGE_MAX_DIMENSION, CLINICAL_IMAGE_MAX_DIMENSION, CLINICAL_IMAGE_COMPRESSION_QUALITY);
                        currentClinicalImagesData.push({ id: uuid(), dataUrl: compressedDataUrl, fileName: capturedFile.name });
                        renderClinicalImageGalleryUI();
                    } catch (error) {
                        console.error("Error processing captured clinical image:", error);
                        showToast("Error processing captured image.", "error");
                    }
                } else {
                    showToast(`Cannot add more than ${MAX_CLINICAL_IMAGES} clinical images.`, 'warning');
                }
            }
        }
        closeCameraModal();
    });
}
// --- END Camera Modal Logic ---


// --- START Treatment & Investigations View ---
function renderTreatmentInvestigationsView(): string {
    const record = currentPatientForTreatment;

    if (!record) {
        return `
            <div class="content-card">
                <h2><i class="fas fa-user-injured"></i> Select Patient for Treatment/Investigations</h2>
                <p>Please select a patient from the records list to manage their treatment and investigations.</p>
                <div class="search-bar">
                    <input type="text" id="treatment-patient-search" placeholder="Search by Patient Name, Patient ID (KAHS...), or Record ID...">
                    <button type="button" id="treatment-patient-search-btn" class="btn btn-primary"><i class="fas fa-search"></i> Search</button>
                </div>
                <div id="treatment-patient-search-results" class="records-table-container mt-3"></div>
            </div>
        `;
    }

    let patientIdentifierString = `Patient ID: ${record.patientId}`;
    if (record.beemaNumber) {
        patientIdentifierString += ` | Beema No.: ${record.beemaNumber}`;
    }
    patientIdentifierString += `, Record ID: ${record.id}`;


    const renderMedicationList = (meds: MedicationEntry[] | undefined, type: 'Antibiotics' | 'Analgesics' | 'OtherMedications') => {
        if (!meds || meds.length === 0) return _internalGenerateNewMedicationEntryHtmlString(type);
        return meds.map(med => `
            <div class="medication-entry" data-med-id="${med.id}">
                 <button type="button" class="remove-medication-btn" data-med-type="${type}" aria-label="Remove medication">&times;</button>
                <div class="form-grid grid-cols-4">
                    <div class="input-group"><label>Drug Name</label><input type="text" name="${type.toLowerCase()}-drugName" value="${med.drugName}" required></div>
                    <div class="input-group"><label>Dose</label><input type="text" name="${type.toLowerCase()}-dose" value="${med.dose}" required></div>
                    <div class="input-group"><label>Route</label><input type="text" name="${type.toLowerCase()}-route" value="${med.route}" required></div>
                    <div class="input-group"><label>Frequency</label><input type="text" name="${type.toLowerCase()}-frequency" value="${med.frequency}" required></div>
                </div>
                <div class="input-group"><label>Duration (Optional)</label><input type="text" name="${type.toLowerCase()}-duration" value="${med.duration || ''}"></div>
            </div>
        `).join('');
    };

    const sortedLabReports = (record.labReportFiles || []).slice().sort((a, b) => {
        const dateA = getComparableDate(a.reportDate);
        const dateB = getComparableDate(b.reportDate);
        return new Date(dateB).getTime() - new Date(dateA).getTime();
    });

    const labReportsHtml = sortedLabReports.map(labEntry => `
        <div class="lab-report-entry-item content-card" data-lab-entry-id="${labEntry.id}">
            <div class="lab-report-header">
                <strong>Report Date:</strong> ${labEntry.reportDate || 'N/A'}
                <span style="float:right;">(Added: ${new Date(labEntry.createdAt).toLocaleDateString()})</span>
            </div>
            <p><strong>File:</strong> <a href="${labEntry.fileUrl}" target="_blank" title="View ${labEntry.fileName}">${labEntry.fileName}</a> (${labEntry.fileType})</p>
            ${labEntry.labParameters && Object.keys(labEntry.labParameters).length > 0 ? `
                <div class="lab-parameters-display">
                    <strong>Lab Parameters:</strong>
                    <ul>
                        ${Object.entries(labEntry.labParameters).map(([key, value]) => `<li><strong>${key}:</strong> ${value}</li>`).join('')}
                    </ul>
                </div>
            ` : '<p><em>No structured lab parameters extracted.</em></p>'}
            <div class="lab-report-actions">
                ${labEntry.rawOcrText ? `<button type="button" class="view-lab-ocr-btn btn btn-info btn-sm" data-lab-entry-id="${labEntry.id}"><i class="fas fa-file-alt"></i> View OCR Text</button>` : ''}
                <button type="button" class="remove-lab-entry-btn btn btn-danger-outline btn-sm" data-lab-entry-id="${labEntry.id}"><i class="fas fa-trash"></i> Remove Report</button>
            </div>
        </div>
    `).join('');


    return `
        <form id="treatment-investigations-form" class="content-card">
            <h2><i class="fas fa-notes-medical"></i> Treatment & Investigations for ${record.patientName} (${patientIdentifierString})</h2>
            <div class="content-card">
                <h3>Conservative Treatment</h3>
                <div class="input-group">
                    <label for="conservativeTreatmentGiven">Conservative Treatment Given (e.g., POP, Slab, Brace)</label>
                    <textarea id="conservativeTreatmentGiven" name="conservativeTreatmentGiven" rows="3">${record.conservativeTreatmentGiven || ''}</textarea>
                </div>
            </div>
            <div class="content-card">
                <h3>Operative Management</h3>
                <div class="form-grid grid-cols-2">
                    <div class="input-group"><label for="operativeDateOfSurgery">Date of Surgery (AD)</label><input type="date" id="operativeDateOfSurgery" name="operativeDateOfSurgery" value="${record.operativeDateOfSurgery || ''}"></div>
                    <div class="input-group"><label for="operativeNameOfSurgery">Name of Surgery</label><input type="text" id="operativeNameOfSurgery" name="operativeNameOfSurgery" value="${record.operativeNameOfSurgery || ''}"></div>
                    <div class="input-group"><label for="operativeApproach">Approach</label><input type="text" id="operativeApproach" name="operativeApproach" value="${record.operativeApproach || ''}"></div>
                    <div class="input-group"><label for="operativeImplantUsed">Implant Used</label><input type="text" id="operativeImplantUsed" name="operativeImplantUsed" value="${record.operativeImplantUsed || ''}"></div>
                </div>
                <div class="input-group"><label for="operativeNotes">Operative Notes</label><textarea id="operativeNotes" name="operativeNotes" rows="4">${record.operativeNotes || ''}</textarea></div>
            </div>
            <div class="content-card">
                <h3>Investigations</h3>
                <div class="input-group">
                    <label for="radiologicalInvestigationDetails">Radiological Investigation Details (X-ray, CT, MRI findings)</label>
                    <textarea id="radiologicalInvestigationDetails" name="radiologicalInvestigationDetails" rows="3">${record.radiologicalInvestigationDetails || ''}</textarea>
                </div>

                <h4>Lab Reports</h4>
                <p class="instruction-text">Upload lab report files (PDF, JPG, PNG). Files >5MB may be compressed (images only). OCR will attempt to extract date and parameters.</p>
                <input type="file" id="labReportFilesInput" multiple accept=".pdf,.jpg,.jpeg,.png" style="display:none;">
                <button type="button" id="uploadLabReportBtn" class="btn btn-secondary btn-sm"><i class="fas fa-upload"></i> Upload Lab Report(s)</button>
                <div id="lab-ocr-indicator" class="ocr-indicator" style="display: none;"><div class="spinner spinner-inline"></div><span>Processing Lab Report OCR...</span></div>

                <div id="lab-reports-list-container" class="mt-3">
                    ${labReportsHtml || '<p>No lab reports uploaded yet.</p>'}
                </div>

                 <div class="input-group mt-2">
                    <label for="manualLabNotes">Manual Lab Notes / Summary</label>
                    <textarea id="manualLabNotes" name="manualLabNotes" rows="3" placeholder="Manually enter key lab findings or summaries here.">${record.manualLabNotes || ''}</textarea>
                </div>
            </div>
            <div class="content-card">
                <h3>Final Diagnosis (Post-treatment)</h3>
                <div class="input-group"><label for="finalDiagnosisTreatment">Final Diagnosis</label><input type="text" id="finalDiagnosisTreatment" name="finalDiagnosisTreatment" value="${record.finalDiagnosisTreatment || record.provisionalDiagnosis || ''}"></div>
            </div>
            <div class="content-card">
                <h3>Condition & Vitals at Discharge</h3>
                <div class="form-grid grid-cols-2">
                    <div class="input-group"><label for="dischargeConditionOfWound">Condition of Wound</label><input type="text" id="dischargeConditionOfWound" name="dischargeConditionOfWound" value="${record.dischargeConditionOfWound || ''}"></div>
                     <div class="input-group"><label for="dischargePulse">Pulse (bpm)</label><input type="text" id="dischargePulse" name="dischargePulse" value="${record.dischargePulse || ''}"></div>
                    <div class="input-group"><label for="dischargeBloodPressure">Blood Pressure (mmHg)</label><input type="text" id="dischargeBloodPressure" name="dischargeBloodPressure" value="${record.dischargeBloodPressure || ''}"></div>
                    <div class="input-group"><label for="dischargeTemperature">Temperature (°F/°C)</label><input type="text" id="dischargeTemperature" name="dischargeTemperature" value="${record.dischargeTemperature || ''}"></div>
                    <div class="input-group"><label for="dischargeRespiratoryRate">Respiratory Rate (breaths/min)</label><input type="text" id="dischargeRespiratoryRate" name="dischargeRespiratoryRate" value="${record.dischargeRespiratoryRate || ''}"></div>
                </div>
            </div>
            <div class="content-card">
                <h3>Medications on Discharge</h3>
                <div class="medication-group"><h4>Antibiotics</h4><div id="dischargeAntibioticsList">${renderMedicationList(record.dischargeAntibiotics, 'Antibiotics')}</div><button type="button" class="btn btn-secondary btn-sm mt-2 add-medication-btn" data-type="Antibiotics"><i class="fas fa-plus"></i> Add Antibiotic</button></div>
                <div class="medication-group"><h4>Analgesics</h4><div id="dischargeAnalgesicsList">${renderMedicationList(record.dischargeAnalgesics, 'Analgesics')}</div><button type="button" class="btn btn-secondary btn-sm mt-2 add-medication-btn" data-type="Analgesics"><i class="fas fa-plus"></i> Add Analgesic</button></div>
                <div class="medication-group"><h4>Other Medications</h4><div id="dischargeOtherMedicationsList">${renderMedicationList(record.dischargeOtherMedications, 'OtherMedications')}</div><button type="button" class="btn btn-secondary btn-sm mt-2 add-medication-btn" data-type="OtherMedications"><i class="fas fa-plus"></i> Add Other Medication</button></div>
            </div>
            <div class="content-card">
                <h3>Advice on Discharge</h3>
                <div class="form-grid grid-cols-2">
                    <div class="input-group"><label for="dischargeDietaryAdvice">Dietary Advice</label><input type="text" id="dischargeDietaryAdvice" name="dischargeDietaryAdvice" value="${record.dischargeDietaryAdvice || ''}"></div>
                    <div class="input-group"><label for="dischargeWoundCareAdvice">Wound Care Advice</label><input type="text" id="dischargeWoundCareAdvice" name="dischargeWoundCareAdvice" value="${record.dischargeWoundCareAdvice || ''}"></div>
                    <div class="input-group"><label for="dischargeDateSutureOut">Date of Suture Out (AD)</label><input type="date" id="dischargeDateSutureOut" name="dischargeDateSutureOut" value="${record.dischargeDateSutureOut || ''}"></div>
                    <div class="input-group"><label for="dischargeNextOpdVisit">Next OPD Visit (AD)</label><input type="date" id="dischargeNextOpdVisit" name="dischargeNextOpdVisit" value="${record.dischargeNextOpdVisit || ''}"></div>
                </div>
                <div class="input-group"><label for="dischargeDressingAdvice">Dressing Advice</label><textarea id="dischargeDressingAdvice" name="dischargeDressingAdvice" rows="2">${record.dischargeDressingAdvice || ''}</textarea></div>
            </div>
            <div class="content-card">
                <h3>Physiotherapy & Rehabilitation</h3>
                 <div class="input-group"><label for="physiotherapyRehabProtocol">Physiotherapy / Rehabilitation Protocol</label><textarea id="physiotherapyRehabProtocol" name="physiotherapyRehabProtocol" rows="3">${record.physiotherapyRehabProtocol || ''}</textarea></div>
                <div class="form-grid grid-cols-3">
                    <div class="input-group"><label for="weightBearingAdvice">Weight Bearing Advice</label><input type="text" id="weightBearingAdvice" name="weightBearingAdvice" value="${record.weightBearingAdvice || ''}"></div>
                    <div class="input-group"><label for="exerciseProtocol">Exercise Protocol</label><input type="text" id="exerciseProtocol" name="exerciseProtocol" value="${record.exerciseProtocol || ''}"></div>
                    <div class="input-group"><label for="restLimbElevationAdvice">Rest & Limb Elevation</label><input type="text" id="restLimbElevationAdvice" name="restLimbElevationAdvice" value="${record.restLimbElevationAdvice || ''}"></div>
                </div>
            </div>
            <div class="form-actions">
                 <button type="button" id="view-discharge-summary-btn" class="btn btn-info"><i class="fas fa-file-alt"></i> View Discharge Summary</button>
                <button type="submit" class="btn btn-primary"><i class="fas fa-save"></i> Save Treatment Details</button>
            </div>
        </form>
    `;
}

function reRenderLabReportsList() {
    const container = S<HTMLDivElement>('#lab-reports-list-container');
    if (!container || !currentPatientForTreatment) return;

    const sortedLabReports = (currentPatientForTreatment.labReportFiles || []).slice().sort((a, b) => {
        const dateA = getComparableDate(a.reportDate);
        const dateB = getComparableDate(b.reportDate);
        return new Date(dateB).getTime() - new Date(dateA).getTime();
    });

    if (sortedLabReports.length === 0) {
        container.innerHTML = '<p>No lab reports uploaded yet.</p>';
        return;
    }

    container.innerHTML = sortedLabReports.map(labEntry => `
        <div class="lab-report-entry-item content-card" data-lab-entry-id="${labEntry.id}">
            <div class="lab-report-header">
                <strong>Report Date:</strong> ${labEntry.reportDate || 'N/A'}
                <span style="float:right;">(Added: ${new Date(labEntry.createdAt).toLocaleDateString()})</span>
            </div>
            <p><strong>File:</strong>
                <a href="${labEntry.fileUrl}" target="_blank" title="View ${labEntry.fileName}">
                    <i class="fas fa-${labEntry.fileType === 'application/pdf' ? 'file-pdf' : 'file-image'}" style="margin-right: 5px;"></i>${labEntry.fileName}
                </a>
            </p>
            ${labEntry.labParameters && Object.keys(labEntry.labParameters).length > 0 ? `
                <div class="lab-parameters-display">
                    <strong>Lab Parameters:</strong>
                    <ul>
                        ${Object.entries(labEntry.labParameters).map(([key, value]) => `<li><strong>${key}:</strong> ${value}</li>`).join('')}
                    </ul>
                </div>
            ` : '<p><em>No structured lab parameters extracted.</em></p>'}
            <div class="lab-report-actions">
                ${labEntry.rawOcrText ? `<button type="button" class="view-lab-ocr-btn btn btn-info btn-sm" data-lab-entry-id="${labEntry.id}"><i class="fas fa-file-alt"></i> View OCR Text</button>` : ''}
                <button type="button" class="remove-lab-entry-btn btn btn-danger-outline btn-sm" data-lab-entry-id="${labEntry.id}"><i class="fas fa-trash"></i> Remove Report</button>
            </div>
        </div>
    `).join('');
}


function attachTreatmentInvestigationsListeners() {
    const form = S<HTMLFormElement>('#treatment-investigations-form');
    const searchInput = S<HTMLInputElement>('#treatment-patient-search');
    const searchBtn = S<HTMLButtonElement>('#treatment-patient-search-btn');
    const searchResultsDiv = S<HTMLDivElement>('#treatment-patient-search-results');

    if (searchInput && searchBtn && searchResultsDiv) {
        const performSearch = () => {
            const searchTerm = searchInput.value.toLowerCase();
            if (!searchTerm) {
                searchResultsDiv.innerHTML = '<p>Enter a name, Patient ID, or Record ID to search.</p>';
                return;
            }
            const results = traumaRecords.filter(r =>
                r.patientName.toLowerCase().includes(searchTerm) ||
                r.id.toLowerCase().includes(searchTerm) ||
                r.patientId.toLowerCase().includes(searchTerm) ||
                (r.beemaNumber && r.beemaNumber.toLowerCase().includes(searchTerm))
            );
            renderTreatmentPatientSearchResults(results, searchResultsDiv);
        };
        searchBtn.addEventListener('click', performSearch);
        searchInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') performSearch(); });
    }

    if (!currentPatientForTreatment) return;

    if (form) {
        form.addEventListener('submit', handleTreatmentInvestigationsFormSubmit);

        const labReportFilesInput = S<HTMLInputElement>('#labReportFilesInput');
        const uploadLabReportBtn = S<HTMLButtonElement>('#uploadLabReportBtn');
        const labReportsListContainer = S<HTMLDivElement>('#lab-reports-list-container');

        uploadLabReportBtn?.addEventListener('click', () => labReportFilesInput?.click());

        labReportFilesInput?.addEventListener('change', async (event) => {
            const files = (event.target as HTMLInputElement).files;
            if (!files || !currentPatientForTreatment) return;
             if (!derivedEncryptionKey) {
                showToast("Warning: Encryption key not available. Cannot securely process lab reports.", "warning");
                if (labReportFilesInput) labReportFilesInput.value = '';
                return;
            }


            if (!currentPatientForTreatment.labReportFiles) {
                currentPatientForTreatment.labReportFiles = [];
            }
            const labOcrIndicator = S<HTMLDivElement>('#lab-ocr-indicator');


            for (const originalFile of Array.from(files)) {
                if(labOcrIndicator) labOcrIndicator.style.display = 'flex';
                const tempDisplayId = uuid();
                const tempItemHtml = `<div class="lab-report-entry-item content-card" data-temp-id="${tempDisplayId}"><i class="fas fa-spinner fa-spin"></i> Processing ${originalFile.name}...</div>`;
                if (labReportsListContainer) {
                    if (labReportsListContainer.innerHTML.includes('<p>No lab reports uploaded yet.</p>')) {
                        labReportsListContainer.innerHTML = tempItemHtml;
                    } else {
                        labReportsListContainer.insertAdjacentHTML('afterbegin', tempItemHtml);
                    }
                }


                let fileForOcr = originalFile;
                let base64Url = '';

                if (originalFile.type.startsWith('image/') && originalFile.size > LARGE_FILE_THRESHOLD_BYTES) {
                    showToast(`Lab report image ${originalFile.name} is large, attempting compression...`, "info");
                    try {
                        base64Url = await compressImage(originalFile, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_MAX_DIM, LARGE_FILE_OCR_COMPRESSION_QUALITY);
                        fileForOcr = await base64StringToFile(base64Url, originalFile.name, originalFile.type);
                        showToast(`Compression for ${originalFile.name} successful.`, "success");
                    } catch (e) {
                        console.error(`Compression error for ${originalFile.name}:`,e);
                        showToast(`Compression failed for ${originalFile.name}. Using original.`, "warning");
                        base64Url = await fileToDataURL(originalFile);
                    }
                } else {
                     base64Url = await fileToDataURL(originalFile);
                }
                 if ((originalFile.type === "application/pdf" || fileForOcr.type === "application/pdf") && fileForOcr.size > 10 * 1024 * 1024) {
                    showToast(`PDF Lab report "${fileForOcr.name}" is very large. OCR might be slow or fail.`, "warning");
                }


                const ocrResult = await performOcr(fileForOcr, 'lab_report');
                if(labOcrIndicator) labOcrIndicator.style.display = 'none';

                const newLabEntry: LabReportEntry = {
                    id: uuid(),
                    fileName: originalFile.name,
                    fileUrl: base64Url,
                    fileType: fileForOcr.type,
                    reportDate: ocrResult.extractedLabData?.reportDate,
                    labParameters: ocrResult.extractedLabData?.labParameters,
                    rawOcrText: ocrResult.rawText,
                    source: 'auto-captured',
                    createdAt: new Date().toISOString()
                };

                currentPatientForTreatment.labReportFiles.push(newLabEntry);

                const tempItemElement = labReportsListContainer?.querySelector(`[data-temp-id="${tempDisplayId}"]`);
                tempItemElement?.remove();
                reRenderLabReportsList();


                if (ocrResult.error) {
                    showToast(`OCR for ${originalFile.name}: ${ocrResult.error}`, 'warning');
                } else if (!ocrResult.extractedLabData?.reportDate && !ocrResult.extractedLabData?.labParameters) {
                    showToast(`OCR for ${originalFile.name} completed, but no structured data (date/parameters) could be extracted. Review raw text.`, 'info');
                } else {
                    showToast(`Lab report ${originalFile.name} processed.`, 'success');
                }
            }
            if (labReportFilesInput) labReportFilesInput.value = '';
        });

        labReportsListContainer?.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            const labEntryId = target.closest<HTMLButtonElement>('.remove-lab-entry-btn, .view-lab-ocr-btn')?.dataset.labEntryId;

            if (!labEntryId || !currentPatientForTreatment || !currentPatientForTreatment.labReportFiles) return;

            const labEntry = currentPatientForTreatment.labReportFiles.find(entry => entry.id === labEntryId);
            if (!labEntry) return;

            if (target.closest('.remove-lab-entry-btn')) {
                currentPatientForTreatment.labReportFiles = currentPatientForTreatment.labReportFiles.filter(entry => entry.id !== labEntryId);
                reRenderLabReportsList();
                encryptAndSaveAllData();
                showToast(`Lab report ${labEntry.fileName} removed.`, 'info');
            } else if (target.closest('.view-lab-ocr-btn')) {
                if (labEntry.rawOcrText) {
                    openModal(`OCR Text - ${labEntry.fileName}`, `<pre style="white-space: pre-wrap; word-wrap: break-word;">${escapeHtml(labEntry.rawOcrText)}</pre>`, 'lg', false);
                } else {
                    showToast("No OCR text available for this report.", "info");
                }
            }
        });

        const addMedicationButtons = SAll<HTMLButtonElement>('.add-medication-btn');
        addMedicationButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const medType = btn.dataset.type as 'Antibiotics' | 'Analgesics' | 'OtherMedications';
                const listId = `discharge${medType}List`;
                const listDiv = S<HTMLDivElement>(`#${listId}`);
                const tempContainer = document.createElement('div');
                tempContainer.innerHTML = _internalGenerateNewMedicationEntryHtmlString(medType);
                const newEntryElement = tempContainer.firstElementChild as HTMLElement;
                if (newEntryElement) { listDiv?.appendChild(newEntryElement); }
            });
        });

        form.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            if (target.classList.contains('remove-medication-btn')) {
                target.closest('.medication-entry')?.remove();
            }
        });

        const viewDischargeSummaryBtn = S<HTMLButtonElement>('#view-discharge-summary-btn');
        viewDischargeSummaryBtn?.addEventListener('click', () => {
             if (currentPatientForTreatment) {
                generateAndShowDischargeSummaryModal(currentPatientForTreatment);
            } else {
                showToast("No patient selected to view discharge summary.", "error");
            }
        });
        if (currentPatientForTreatment && currentPatientForTreatment.labReportFiles) {
            reRenderLabReportsList();
        }
    }
}
function escapeHtml(unsafe: string): string {
    if (!unsafe) return '';
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }

function renderTreatmentPatientSearchResults(results: TraumaRecord[], container: HTMLDivElement) {
    if (results.length === 0) {
        container.innerHTML = '<p>No matching records found.</p>';
        return;
    }
    container.innerHTML = `
        <table class="records-table">
            <thead><tr><th>Record ID</th><th>Patient ID</th><th>Beema No.</th><th>Name</th><th>Age</th><th>Sex</th><th>Action</th></tr></thead>
            <tbody>
                ${results.map(r => `
                    <tr>
                        <td>${r.id}</td><td>${r.patientId}</td><td>${r.beemaNumber || 'N/A'}</td><td>${r.patientName}</td><td>${formatAge(r.age)}</td><td>${r.sex}</td>
                        <td><button class="btn btn-sm btn-success select-patient-for-treatment-btn" data-record-id="${r.id}"><i class="fas fa-check-circle"></i> Select</button></td>
                    </tr>`).join('')}
            </tbody>
        </table>`;
    SAll<HTMLButtonElement>('.select-patient-for-treatment-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const recordId = btn.dataset.recordId;
            const selectedRecord = traumaRecords.find(r => r.id === recordId);
            if (selectedRecord) {
                currentPatientForTreatment = selectedRecord;
                navigateTo('treatment-investigations');
            }
        });
    });
}

async function handleTreatmentInvestigationsFormSubmit(event: Event) {
    event.preventDefault();
    if (!currentPatientForTreatment) {
        showToast("No patient selected for treatment.", "error");
        return;
    }
    if (!derivedEncryptionKey) {
        showToast("Warning: Encryption key not available. Cannot save treatment data securely.", "warning");
        return;
    }

    const form = event.target as HTMLFormElement;
    const formData = new FormData(form);
    const getMedicationData = (type: 'Antibiotics' | 'Analgesics' | 'OtherMedications'): MedicationEntry[] => {
        const meds: MedicationEntry[] = [];
        const medEntries = form.querySelectorAll<HTMLDivElement>(`#discharge${type}List .medication-entry`);
        medEntries.forEach(entry => {
            const drugName = entry.querySelector<HTMLInputElement>(`input[name="${type.toLowerCase()}-drugName"]`)?.value;
            const dose = entry.querySelector<HTMLInputElement>(`input[name="${type.toLowerCase()}-dose"]`)?.value;
            const route = entry.querySelector<HTMLInputElement>(`input[name="${type.toLowerCase()}-route"]`)?.value;
            const frequency = entry.querySelector<HTMLInputElement>(`input[name="${type.toLowerCase()}-frequency"]`)?.value;
            const duration = entry.querySelector<HTMLInputElement>(`input[name="${type.toLowerCase()}-duration"]`)?.value;
            if (drugName && dose && route && frequency) {
                meds.push({ id: entry.dataset.medId || uuid(), drugName, dose, route, frequency, duration: duration || undefined });
            }
        });
        return meds;
    };
    const updatedRecordData: Partial<TraumaRecord> = {
        conservativeTreatmentGiven: formData.get('conservativeTreatmentGiven') as string || undefined,
        operativeDateOfSurgery: formData.get('operativeDateOfSurgery') as string || undefined,
        operativeNameOfSurgery: formData.get('operativeNameOfSurgery') as string || undefined,
        operativeApproach: formData.get('operativeApproach') as string || undefined,
        operativeImplantUsed: formData.get('operativeImplantUsed') as string || undefined,
        operativeNotes: formData.get('operativeNotes') as string || undefined,
        radiologicalInvestigationDetails: formData.get('radiologicalInvestigationDetails') as string || undefined,
        manualLabNotes: formData.get('manualLabNotes') as string || undefined,
        finalDiagnosisTreatment: formData.get('finalDiagnosisTreatment') as string || undefined,
        dischargeConditionOfWound: formData.get('dischargeConditionOfWound') as string || undefined,
        dischargePulse: formData.get('dischargePulse') as string || undefined,
        dischargeBloodPressure: formData.get('dischargeBloodPressure') as string || undefined,
        dischargeTemperature: formData.get('dischargeTemperature') as string || undefined,
        dischargeRespiratoryRate: formData.get('dischargeRespiratoryRate') as string || undefined,
        dischargeAntibiotics: getMedicationData('Antibiotics'),
        dischargeAnalgesics: getMedicationData('Analgesics'),
        dischargeOtherMedications: getMedicationData('OtherMedications'),
        dischargeDietaryAdvice: formData.get('dischargeDietaryAdvice') as string || undefined,
        dischargeWoundCareAdvice: formData.get('dischargeWoundCareAdvice') as string || undefined,
        dischargeDateSutureOut: formData.get('dischargeDateSutureOut') as string || undefined,
        dischargeNextOpdVisit: formData.get('dischargeNextOpdVisit') as string || undefined,
        dischargeDressingAdvice: formData.get('dischargeDressingAdvice') as string || undefined,
        physiotherapyRehabProtocol: formData.get('physiotherapyRehabProtocol') as string || undefined,
        weightBearingAdvice: formData.get('weightBearingAdvice') as string || undefined,
        exerciseProtocol: formData.get('exerciseProtocol') as string || undefined,
        restLimbElevationAdvice: formData.get('restLimbElevationAdvice') as string || undefined,
    };
    const recordIndex = traumaRecords.findIndex(r => r.id === currentPatientForTreatment!.id);
    if (recordIndex > -1) {
        traumaRecords[recordIndex] = {
            ...traumaRecords[recordIndex],
            ...updatedRecordData,
            labReportFiles: currentPatientForTreatment.labReportFiles, // Ensure these are preserved from potential async uploads
            updatedAt: new Date().toISOString(),
            updatedBy: currentUser!.id,
        };
        currentPatientForTreatment = traumaRecords[recordIndex];
        await encryptAndSaveAllData();
        showToast('Treatment details saved successfully!', 'success');
    } else {
        showToast("Error: Could not find the patient record to update.", "error");
    }
}
// --- END Treatment & Investigations View ---

// --- START Admin Dashboard ---
function renderAdminDashboard(): string {
    const totalRecords = traumaRecords.length;
    const totalUsers = users.length;
    const activeUsers = users.filter(u => u.isActive).length;
    const newOcrErrors = adminErrorLog.filter(log => log.status === 'new').length;

    const usersHtml = users.map(user => `
        <tr>
            <td>${escapeHtml(user.id)}</td>
            <td>${escapeHtml(user.username)}</td>
            <td>${escapeHtml(user.role)}</td>
            <td>
                <span class="status-badge status-${user.isActive ? 'active' : 'inactive'}">
                    ${user.isActive ? 'Active' : 'Inactive'}
                </span>
            </td>
            <td>${new Date(user.createdAt).toLocaleDateString()}</td>
            <td>
                ${currentUser && user.id === currentUser.id ? 
                    '<button class="btn btn-sm btn-secondary" disabled>Self</button>' :
                    (user.isActive ? 
                        `<button class="btn btn-sm btn-warning deactivate-user-btn" data-user-id="${user.id}"><i class="fas fa-user-times"></i> Deactivate</button>` :
                        `<button class="btn btn-sm btn-success activate-user-btn" data-user-id="${user.id}"><i class="fas fa-user-check"></i> Activate</button>`
                    )
                }
            </td>
        </tr>
    `).join('');

    return `
        <div class="content-card">
            <h2><i class="fas fa-user-shield"></i> Admin Dashboard</h2>
            
            <div class="admin-stats-grid">
                <div class="stat-card">
                    <i class="fas fa-notes-medical stat-icon"></i>
                    <div class="stat-value">${totalRecords}</div>
                    <div class="stat-label">Total Trauma Records</div>
                </div>
                <div class="stat-card">
                    <i class="fas fa-users stat-icon"></i>
                    <div class="stat-value">${activeUsers} / ${totalUsers}</div>
                    <div class="stat-label">Active / Total Users</div>
                </div>
                <div class="stat-card">
                    <i class="fas fa-exclamation-triangle stat-icon"></i>
                    <div class="stat-value">${newOcrErrors}</div>
                    <div class="stat-label">New OCR Errors</div>
                </div>
                 <div class="stat-card">
                    <i class="fas fa-key stat-icon"></i>
                    <div class="stat-value">${derivedEncryptionKey ? 'Active' : 'Inactive'}</div>
                    <div class="stat-label">Encryption Key Status</div>
                </div>
            </div>

            <div class="content-card mt-3">
                <h3>User Management</h3>
                <div class="records-table-container">
                    <table class="records-table" id="admin-user-management-table">
                        <thead>
                            <tr>
                                <th>User ID</th>
                                <th>Username</th>
                                <th>Role</th>
                                <th>Status</th>
                                <th>Created At</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${usersHtml}
                        </tbody>
                    </table>
                </div>
            </div>
             <div class="content-card mt-3">
                <h3>Quick Actions</h3>
                 <button id="admin-view-all-records-btn" class="btn btn-info"><i class="fas fa-list-ul"></i> View All Records</button>
                 <button id="admin-add-new-record-btn" class="btn btn-primary"><i class="fas fa-user-plus"></i> Add New Patient Record</button>
            </div>
        </div>
    `;
}

function attachAdminDashboardListeners() {
    const userTable = S<HTMLTableElement>('#admin-user-management-table');
    userTable?.addEventListener('click', async (event) => {
        const target = event.target as HTMLElement;
        const button = target.closest<HTMLButtonElement>('.activate-user-btn, .deactivate-user-btn');
        if (!button) return;

        const userIdToToggle = button.dataset.userId;
        if (!userIdToToggle) return;

        if (currentUser && userIdToToggle === currentUser.id) {
            showToast("You cannot change the status of your own account.", "warning");
            return;
        }

        const userToUpdate = users.find(u => u.id === userIdToToggle);
        if (userToUpdate) {
            userToUpdate.isActive = !userToUpdate.isActive;
            await encryptAndSaveAllData();
            showToast(`User ${userToUpdate.username} status changed to ${userToUpdate.isActive ? 'Active' : 'Inactive'}.`, "success");
            // Re-render the dashboard to reflect changes
            const appContent = S<HTMLElement>('#app-content');
            if (appContent && currentView === 'admin-dashboard') {
                appContent.innerHTML = renderAdminDashboard();
                attachAdminDashboardListeners(); // Re-attach listeners
            }
        } else {
            showToast("Error: User not found.", "error");
        }
    });

    S<HTMLButtonElement>('#admin-view-all-records-btn')?.addEventListener('click', () => {
        navigateTo('view-records');
    });
    S<HTMLButtonElement>('#admin-add-new-record-btn')?.addEventListener('click', () => {
        navigateTo('patient-details');
    });
}
// --- END Admin Dashboard ---

// --- START OCR Error Log View ---
function renderOcrErrorLogView(): string {
    const errorLogHtml = adminErrorLog.length > 0
        ? `<table class="records-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Type</th>
                    <th>Message</th>
                    <th>File Name</th>
                    <th>Patient ID</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${adminErrorLog.map(log => `
                    <tr class="${log.status === 'resolved' ? 'resolved-log' : ''}">
                        <td>${new Date(log.timestamp).toLocaleString()}</td>
                        <td>${log.errorType}</td>
                        <td class="error-message-cell">${escapeHtml(log.message)}</td>
                        <td>${log.fileName || 'N/A'}</td>
                        <td>${log.patientId || 'N/A'}</td>
                        <td><span class="status-badge status-${log.status}">${log.status}</span></td>
                        <td>
                            ${log.status === 'new' ? `<button class="btn btn-sm btn-success resolve-error-btn" data-log-id="${log.id}"><i class="fas fa-check"></i> Mark Resolved</button>` : ''}
                            ${(log.errorType.includes('OCR_FAILURE') || log.errorType.includes('PATIENT_DOC_OCR_FAILURE')) && log.originalFile ? `<button class="btn btn-sm btn-warning retry-ocr-btn" data-log-id="${log.id}"><i class="fas fa-redo"></i> Retry OCR</button>` : ''}
                        </td>
                    </tr>`).join('')}
            </tbody>
           </table>`
        : '<p>No OCR errors logged.</p>';

    return `
        <div class="content-card">
            <h2><i class="fas fa-exclamation-triangle"></i> OCR Error Log</h2>
            <div id="ocr-error-log-list" class="records-table-container">
                ${errorLogHtml}
            </div>
        </div>
    `;
}

function attachOcrErrorLogListeners() {
    S<HTMLDivElement>('#ocr-error-log-list')?.addEventListener('click', (event) => {
        const target = event.target as HTMLElement;
        const logId = target.closest<HTMLButtonElement>('.resolve-error-btn, .retry-ocr-btn')?.dataset.logId;
        if (!logId) return;

        const logEntry = adminErrorLog.find(log => log.id === logId);
        if (!logEntry) return;

        if (target.closest('.resolve-error-btn')) {
            logEntry.status = 'resolved';
            encryptAndSaveAllData();
            navigateTo('ocr-error-log'); // Re-render the view
            showToast(`Error log ${logId} marked as resolved.`, 'success');
        } else if (target.closest('.retry-ocr-btn')) {
            if (logEntry.originalFile) {
                ocrErrorToRetry = logEntry;
                // Redirect to patient details form, pre-filling from log entry if possible
                // This is a simplified retry; a more complex one might pass the file directly
                showToast(`Retrying OCR for ${logEntry.fileName}. Please upload the document again on the patient form.`, 'info');
                // For simplicity, just navigate to new patient form.
                // A more advanced retry would involve passing the file or its data for reprocessing.
                if (logEntry.recordId) {
                     navigateTo('patient-details', logEntry.recordId, logEntry);
                } else {
                     navigateTo('patient-details', undefined, logEntry);
                }
            } else {
                showToast("Cannot retry OCR: Original file not available.", "error");
            }
        }
    });
}
// --- END OCR Error Log View ---

// --- START View Records ---
function renderRecordsListView() {
    const appContent = S<HTMLElement>('#app-content');
    if (!appContent) return;

    let recordsHtml = '<p>No records found. Start by adding a new patient.</p>';
    if (traumaRecords.length > 0) {
        recordsHtml = `
            <table class="records-table">
                <thead>
                    <tr>
                        <th>Patient ID</th>
                        <th>Name</th>
                        <th>Age</th>
                        <th>Sex</th>
                        <th>Injury Date (AD)</th>
                        <th>Diagnosis</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${traumaRecords.map(record => `
                        <tr>
                            <td>${record.patientId}</td>
                            <td>${record.patientName}</td>
                            <td>${formatAge(record.age)}</td>
                            <td>${record.sex}</td>
                            <td>${record.dateOfInjuryAD}</td>
                            <td>${record.provisionalDiagnosis}</td>
                            <td>
                                <button class="btn btn-sm btn-info view-record-btn" data-record-id="${record.id}"><i class="fas fa-eye"></i> View</button>
                                <button class="btn btn-sm btn-warning edit-record-btn" data-record-id="${record.id}"><i class="fas fa-edit"></i> Edit</button>
                                ${currentUser?.role === 'admin' ? `<button class="btn btn-sm btn-danger delete-record-btn" data-record-id="${record.id}"><i class="fas fa-trash"></i> Delete</button>` : ''}
                            </td>
                        </tr>`).join('')}
                </tbody>
            </table>
        `;
    }

    appContent.innerHTML = `
        <div class="content-card">
            <h2><i class="fas fa-clipboard-list"></i> View Trauma Records</h2>
            <div id="records-list" class="records-table-container">
                ${recordsHtml}
            </div>
        </div>
    `;
    attachRecordsListViewListeners();
}

function attachRecordsListViewListeners() {
    S<HTMLDivElement>('#records-list')?.addEventListener('click', (event) => {
        const target = event.target as HTMLElement;
        const recordId = target.closest<HTMLButtonElement>('.view-record-btn, .edit-record-btn, .delete-record-btn')?.dataset.recordId;

        if (!recordId) return;

        if (target.closest('.view-record-btn')) {
            const record = traumaRecords.find(r => r.id === recordId);
            if (record) {
                // For now, just re-navigate to patient-details in edit mode (acting as view)
                // A dedicated read-only view could be implemented later.
                navigateTo('patient-details', record.id);
            }
        } else if (target.closest('.edit-record-btn')) {
            navigateTo('patient-details', recordId);
        } else if (target.closest('.delete-record-btn') && currentUser?.role === 'admin') {
            const confirmed = confirm(`Are you sure you want to delete record ${recordId}? This action cannot be undone.`);
            if (confirmed) {
                traumaRecords = traumaRecords.filter(r => r.id !== recordId);
                encryptAndSaveAllData();
                renderRecordsListView(); // Re-render
                showToast(`Record ${recordId} deleted.`, 'success');
            }
        }
    });
}
// --- END View Records ---


// --- START DISCHARGE SUMMARY MODAL ---
function generateAndShowDischargeSummaryModal(record: TraumaRecord) {
    if (!record) {
        showToast("Cannot generate discharge summary: No patient record loaded.", "error");
        return;
    }

    let summaryHtml = `
        <div class="discharge-summary-modal">
            <style>
                .discharge-summary-modal { font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; }
                .discharge-summary-modal h3 { font-size: 1.5em; margin-bottom: 10px; color: var(--primary-color); border-bottom: 1px solid #eee; padding-bottom: 5px; }
                .discharge-summary-modal h4 { font-size: 1.2em; margin-top: 15px; margin-bottom: 8px; color: var(--dark-color); }
                .discharge-summary-modal p { margin-bottom: 5px; }
                .discharge-summary-modal strong { color: #333; }
                .discharge-summary-modal ul { list-style-type: disc; margin-left: 20px; padding-left: 0; }
                .discharge-summary-modal .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
                .discharge-summary-modal .section { margin-bottom: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }
            </style>
            <h3>Discharge Summary</h3>
            
            <div class="section">
                <h4>Patient Details</h4>
                <p><strong>Patient Name:</strong> ${escapeHtml(record.patientName)}</p>
                <p><strong>Patient ID:</strong> ${escapeHtml(record.patientId)}</p>
                <p><strong>Age:</strong> ${formatAge(record.age)}</p>
                <p><strong>Sex:</strong> ${escapeHtml(record.sex)}</p>
                ${record.beemaNumber ? `<p><strong>Beema No.:</strong> ${escapeHtml(record.beemaNumber)}</p>` : ''}
                ${record.address ? `<p><strong>Address:</strong> ${escapeHtml(record.address)}</p>` : ''}
                ${record.contactNumber ? `<p><strong>Contact:</strong> ${escapeHtml(record.contactNumber)}</p>` : ''}
            </div>

            <div class="section">
                <h4>Admission & Injury Details</h4>
                <p><strong>Date of Injury:</strong> ${escapeHtml(record.dateOfInjuryAD)} (AD) / ${escapeHtml(record.dateOfInjuryBS)} (BS)</p>
                <p><strong>Time of Injury:</strong> ${escapeHtml(record.timeOfInjury)}</p>
                <p><strong>Mechanism of Injury:</strong> ${escapeHtml(record.modeOfInjury)}${record.otherMOI ? ` (${escapeHtml(record.otherMOI)})` : ''}</p>
                 <p><strong>Provisional Diagnosis:</strong> ${escapeHtml(record.provisionalDiagnosis)}</p>
            </div>
            
            ${record.operativeDateOfSurgery || record.operativeNameOfSurgery ? `
            <div class="section">
                <h4>Operative Management</h4>
                ${record.operativeDateOfSurgery ? `<p><strong>Date of Surgery:</strong> ${escapeHtml(record.operativeDateOfSurgery)}</p>` : ''}
                ${record.operativeNameOfSurgery ? `<p><strong>Name of Surgery:</strong> ${escapeHtml(record.operativeNameOfSurgery)}</p>` : ''}
                ${record.operativeApproach ? `<p><strong>Approach:</strong> ${escapeHtml(record.operativeApproach)}</p>` : ''}
                ${record.operativeImplantUsed ? `<p><strong>Implant Used:</strong> ${escapeHtml(record.operativeImplantUsed)}</p>` : ''}
                ${record.operativeNotes ? `<p><strong>Operative Notes:</strong> ${escapeHtml(record.operativeNotes)}</p>` : ''}
            </div>` : ''}

            ${record.conservativeTreatmentGiven ? `
            <div class="section">
                <h4>Conservative Treatment</h4>
                <p>${escapeHtml(record.conservativeTreatmentGiven)}</p>
            </div>` : ''}
            
            <div class="section">
                <h4>Final Diagnosis</h4>
                <p>${escapeHtml(record.finalDiagnosisTreatment || record.provisionalDiagnosis || 'N/A')}</p>
            </div>

            <div class="section">
                <h4>Condition at Discharge</h4>
                <div class="grid-2">
                    ${record.dischargeConditionOfWound ? `<p><strong>Wound Condition:</strong> ${escapeHtml(record.dischargeConditionOfWound)}</p>` : ''}
                    ${record.dischargePulse ? `<p><strong>Pulse:</strong> ${escapeHtml(record.dischargePulse)} bpm</p>` : ''}
                    ${record.dischargeBloodPressure ? `<p><strong>BP:</strong> ${escapeHtml(record.dischargeBloodPressure)} mmHg</p>` : ''}
                    ${record.dischargeTemperature ? `<p><strong>Temp:</strong> ${escapeHtml(record.dischargeTemperature)}</p>` : ''}
                    ${record.dischargeRespiratoryRate ? `<p><strong>RR:</strong> ${escapeHtml(record.dischargeRespiratoryRate)} /min</p>` : ''}
                </div>
            </div>`;

    const renderMedListForSummary = (meds?: MedicationEntry[], title?: string) => {
        if (!meds || meds.length === 0) return '';
        let html = title ? `<h5>${title}</h5>` : '';
        html += '<ul>';
        meds.forEach(med => {
            html += `<li>${escapeHtml(med.drugName)} - ${escapeHtml(med.dose)} ${escapeHtml(med.route)} ${escapeHtml(med.frequency)} ${med.duration ? ` (for ${escapeHtml(med.duration)})` : ''}</li>`;
        });
        html += '</ul>';
        return html;
    };

    if (record.dischargeAntibiotics?.length || record.dischargeAnalgesics?.length || record.dischargeOtherMedications?.length) {
        summaryHtml += `
            <div class="section">
                <h4>Medications on Discharge</h4>
                ${renderMedListForSummary(record.dischargeAntibiotics, 'Antibiotics:')}
                ${renderMedListForSummary(record.dischargeAnalgesics, 'Analgesics:')}
                ${renderMedListForSummary(record.dischargeOtherMedications, 'Other Medications:')}
            </div>`;
    }

    summaryHtml += `
            <div class="section">
                <h4>Advice on Discharge</h4>
                ${record.dischargeDietaryAdvice ? `<p><strong>Dietary Advice:</strong> ${escapeHtml(record.dischargeDietaryAdvice)}</p>` : ''}
                ${record.dischargeWoundCareAdvice ? `<p><strong>Wound Care:</strong> ${escapeHtml(record.dischargeWoundCareAdvice)}</p>` : ''}
                ${record.dischargeDressingAdvice ? `<p><strong>Dressing Advice:</strong> ${escapeHtml(record.dischargeDressingAdvice)}</p>` : ''}
                ${record.dischargeDateSutureOut ? `<p><strong>Suture Out Date:</strong> ${escapeHtml(record.dischargeDateSutureOut)}</p>` : ''}
                ${record.dischargeNextOpdVisit ? `<p><strong>Next OPD Visit:</strong> ${escapeHtml(record.dischargeNextOpdVisit)}</p>` : ''}
            </div>`;
    
    if (record.physiotherapyRehabProtocol || record.weightBearingAdvice || record.exerciseProtocol || record.restLimbElevationAdvice) {
         summaryHtml += `
            <div class="section">
                <h4>Physiotherapy & Rehabilitation</h4>
                ${record.physiotherapyRehabProtocol ? `<p><strong>Protocol:</strong> ${escapeHtml(record.physiotherapyRehabProtocol)}</p>` : ''}
                ${record.weightBearingAdvice ? `<p><strong>Weight Bearing:</strong> ${escapeHtml(record.weightBearingAdvice)}</p>` : ''}
                ${record.exerciseProtocol ? `<p><strong>Exercise:</strong> ${escapeHtml(record.exerciseProtocol)}</p>` : ''}
                ${record.restLimbElevationAdvice ? `<p><strong>Rest & Elevation:</strong> ${escapeHtml(record.restLimbElevationAdvice)}</p>` : ''}
            </div>`;
    }

    summaryHtml += `</div>`; // Close discharge-summary-modal

    openModal(`Discharge Summary - ${record.patientName}`, summaryHtml, 'lg', { confirmText: 'Print (Simulated)', cancelText: 'Close' });
    
    // Simulate Print functionality if added
    const confirmBtn = S<HTMLButtonElement>('#modal-confirm-button');
    confirmBtn?.addEventListener('click', () => {
        // In a real app, you'd use window.print() or a library
        showToast("Print functionality is simulated. In a real app, this would print the summary.", "info");
        // For now, could try to print just the modal body
        const modalBody = S<HTMLDivElement>('#modal-body');
        if (modalBody) {
             // Basic print (might need more styling for print media queries)
            // const printWindow = window.open('', '_blank');
            // printWindow?.document.write('<html><head><title>Discharge Summary</title>');
            // // Add styles here if needed for print
            // printWindow?.document.write('</head><body>');
            // printWindow?.document.write(modalBody.innerHTML);
            // printWindow?.document.write('</body></html>');
            // printWindow?.document.close();
            // printWindow?.print();
        }
    }, { once: true }); // Ensure listener is only added once
}
// --- END DISCHARGE SUMMARY MODAL ---


// --- START Application Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    S<HTMLFormElement>('#login-form')?.addEventListener('submit', handleLogin);
    S<HTMLButtonElement>('#logout-btn')?.addEventListener('click', handleLogout);
    S<HTMLButtonElement>('#nurse-logout-button')?.addEventListener('click', handleLogout);

    SAll<HTMLAnchorElement>('#app-nav a.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const viewId = (e.target as HTMLAnchorElement).getAttribute('data-view');
            if (viewId) navigateTo(viewId);
        });
    });

    SAll<HTMLAnchorElement>('#nurse-app-nav a.nurse-nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const viewId = (e.target as HTMLAnchorElement).getAttribute('data-nurse-view') as 'patientDocUpload' | 'labReportUpload' | 'viewLabHistory' | null;
            if (viewId) navigateToNurseView(viewId);
        });
    });


    S<HTMLButtonElement>('#modal-cancel-button')?.addEventListener('click', closeModal);
    S<HTMLSpanElement>('.close-button')?.addEventListener('click', closeModal); // For the 'x' in modal header
    S<HTMLDivElement>('#modal-container')?.addEventListener('click', (event) => {
        if ((event.target as HTMLElement).id === 'modal-container') closeModal();
    });

    setupCameraModalListeners();


    window.addEventListener('online', () => { isOffline = false; showToast("Back online!", "success"); });
    window.addEventListener('offline', () => { isOffline = true; showToast("You are offline. Some features may be limited.", "warning"); });

    if (isOffline) {
        showToast("Application started in offline mode. Some features may be limited.", "warning");
    }

    if (!GEMINI_API_KEY) {
        logAdminError('OCR_API_KEY_MISSING', "GEMINI_API_KEY is not configured. OCR functionality will be limited.");
    }

    // Attempt to initialize Tesseract on load, but don't block.
    // User interaction (like clicking OCR button) will also trigger init if not ready.
    if (!isOffline) { // Only try to load Tesseract if online
        initializeTesseractWorker().catch(err => console.error("Initial Tesseract load failed:", err));
    } else {
        showToast("Offline: Local OCR (Tesseract) will not be loaded automatically. It might load if you go online.", "info");
    }

});
// --- END Application Initialization ---
