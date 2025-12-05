import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import * as Print from 'expo-print';

export interface AnalysisResults {
  angle: number;
  height: number;
  length: number;
  interdistance: number;
  ar_value: number;
  angle_image?: string;
  height_image?: string;
  angle_scale_factor?: number;
  height_scale_factor?: number;
}

export interface PdfServiceParams {
  analysisResults: AnalysisResults;
  diameter: number;
  numRows: number;
}

export class PdfService {
  static async generateReport({
    analysisResults,
    diameter,
    numRows,
  }: PdfServiceParams): Promise<void> {
    try {
      // Convert image URIs to base64 data URLs
      const processedResults = {
        ...analysisResults,
        angle_image: analysisResults.angle_image ? await this.convertImageToBase64(analysisResults.angle_image) : undefined,
        height_image: analysisResults.height_image ? await this.convertImageToBase64(analysisResults.height_image) : undefined,
      };

      // Create HTML content for the PDF
      const htmlContent = this.generateHtmlContent({
        analysisResults: processedResults,
        diameter,
        numRows,
      });

      // Generate PDF using expo-print
      const { uri } = await Print.printToFileAsync({
        html: htmlContent,
        base64: false,
      });

      // Share the PDF
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(uri, {
          mimeType: 'application/pdf',
          dialogTitle: 'TMT Bar Analysis Report',
        });
      } else {
        // Fallback: save to documents directory (updated API)
        const fileName = `tmt_analysis_report_${Date.now()}.pdf`;
        const destinationFile = new FileSystem.File(FileSystem.Paths.document, fileName);
        const destinationUri = destinationFile.uri;
        
        await FileSystem.moveAsync({
          from: uri,
          to: destinationUri,
        });
        
        console.log(`PDF saved to: ${destinationUri}`);
      }
    } catch (error) {
      console.error('Error generating PDF:', error);
      throw new Error(`Failed to generate PDF: ${error}`);
    }
  }

  // Convert image URI to base64 data URL
  private static async convertImageToBase64(imageUri: string): Promise<string> {
    try {
      // Read the image file
      const base64 = await FileSystem.readAsStringAsync(imageUri, {
        encoding: 'base64',
      });
      
      // Determine the image type from the URI
      const imageType = this.getImageTypeFromUri(imageUri);
      
      // Return as data URL
      return `data:${imageType};base64,${base64}`;
    } catch (error) {
      console.error('Error converting image to base64:', error);
      // Return a placeholder or empty string if conversion fails
      return '';
    }
  }

  // Get image MIME type from URI
  private static getImageTypeFromUri(uri: string): string {
    if (uri.includes('.jpg') || uri.includes('.jpeg')) {
      return 'image/jpeg';
    } else if (uri.includes('.png')) {
      return 'image/png';
    } else if (uri.includes('.webp')) {
      return 'image/webp';
    } else {
      // Default to JPEG
      return 'image/jpeg';
    }
  }

  private static generateHtmlContent({
    analysisResults,
    diameter,
    numRows,
  }: PdfServiceParams): string {
    const currentDate = new Date().toLocaleString();
    
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>TMT Bar Analysis Report</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 40px;
            color: #333;
            line-height: 1.6;
          }
          .header {
            text-align: center;
            border-bottom: 3px solid #1A73E8;
            padding-bottom: 20px;
            margin-bottom: 30px;
          }
          .header h1 {
            color: #1A73E8;
            font-size: 28px;
            margin: 0;
            font-weight: bold;
          }
          .header .subtitle {
            color: #666;
            font-size: 16px;
            margin-top: 10px;
          }
          .section {
            margin-bottom: 30px;
          }
          .section h2 {
            color: #1A73E8;
            font-size: 20px;
            border-bottom: 2px solid #E8F0FE;
            padding-bottom: 8px;
            margin-bottom: 20px;
          }
          .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
          }
          .results-table th {
            background-color: #1A73E8;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
          }
          .results-table td {
            padding: 12px;
            border-bottom: 1px solid #E0E0E0;
          }
          .results-table tr:nth-child(even) {
            background-color: #F8F9FA;
          }
          .parameter-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
          }
          .parameter-card {
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1A73E8;
          }
          .parameter-label {
            font-weight: bold;
            color: #1A73E8;
            margin-bottom: 8px;
          }
          .parameter-value {
            font-size: 18px;
            color: #333;
          }
          .image-section {
            margin-top: 30px;
          }
          .image-container {
            margin-bottom: 20px;
            text-align: center;
            padding: 15px;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            background-color: #FAFAFA;
          }
          .image-container h3 {
            color: #1A73E8;
            margin-bottom: 10px;
            font-size: 18px;
          }
          .analysis-image {
            max-width: 100%;
            max-height: 400px;
            height: auto;
            border: 2px solid #E0E0E0;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
          }
          .image-description {
            color: #666;
            font-size: 14px;
            margin: 10px 0;
            font-style: italic;
          }
          .image-details {
            margin-top: 15px;
            text-align: left;
            background-color: #F8F9FA;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #1A73E8;
          }
          .image-details p {
            margin: 5px 0;
            font-size: 12px;
            color: #333;
          }
          .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #E0E0E0;
            text-align: center;
            color: #666;
            font-size: 12px;
          }
          .ar-value {
            background-color: #E8F5E8;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #34A853;
            margin-top: 20px;
          }
          .ar-value-label {
            font-weight: bold;
            color: #34A853;
            margin-bottom: 5px;
          }
          .ar-value-number {
            font-size: 24px;
            color: #333;
            font-weight: bold;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>TATA Steel TMT Bar Analysis Report</h1>
          <div class="subtitle">Professional Analysis Results</div>
        </div>

        <div class="section">
          <h2>Analysis Results</h2>
          <table class="results-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Rib Angle</td>
                <td>${analysisResults.angle.toFixed(2)}°</td>
              </tr>
              <tr>
                <td>Rib Height</td>
                <td>${analysisResults.height.toFixed(2)} mm</td>
              </tr>
              <tr>
                <td>Rib Length</td>
                <td>${analysisResults.length.toFixed(2)} mm</td>
              </tr>
              <tr>
                <td>Interdistance</td>
                <td>${analysisResults.interdistance.toFixed(2)} mm</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="section">
          <h2>AR Value</h2>
          <div class="ar-value">
            <div class="ar-value-label">Calculated AR Value</div>
            <div class="ar-value-number">${analysisResults.ar_value.toFixed(3)}</div>
          </div>
        </div>

        <div class="section">
          <h2>Analysis Parameters</h2>
          <div class="parameter-grid">
            <div class="parameter-card">
              <div class="parameter-label">Bar Diameter</div>
              <div class="parameter-value">${diameter.toFixed(1)} mm</div>
            </div>
            <div class="parameter-card">
              <div class="parameter-label">Number of Rows</div>
              <div class="parameter-value">${numRows}</div>
            </div>
          </div>
        </div>

        <div class="section">
          <h2>Calibration Information</h2>
          <div class="parameter-card">
            <div class="parameter-label">Scale Factor Explanation</div>
            <div class="parameter-value">
              <p>The scale factor represents the number of pixels per millimeter in the captured images. This is calculated by measuring the known bar diameter against the overlay size in pixels.</p>
              <p><strong>Formula:</strong> Scale Factor = Overlay Size (pixels) ÷ Bar Diameter (mm)</p>
              <p><strong>Higher scale factor = More accurate measurements</strong></p>
              <p><strong>Measurement accuracy:</strong> ±(1/scale_factor) mm per pixel</p>
            </div>
          </div>
        </div>

        ${analysisResults.angle_image || analysisResults.height_image ? `
        <div class="section image-section">
          <h2>Analysis Images</h2>
          ${analysisResults.angle_image ? `
          <div class="image-container">
            <h3>📐 Angle & Length Analysis</h3>
            <p class="image-description">Image used for measuring rib angle, length, and interdistance</p>
            <img src="${analysisResults.angle_image}" alt="Angle Analysis" class="analysis-image" />
            <div class="image-details">
              <p><strong>Analysis Type:</strong> Angle & Length Measurement</p>
              <p><strong>Scale Factor:</strong> ${analysisResults.angle_scale_factor ? analysisResults.angle_scale_factor.toFixed(2) + ' pixels/mm' : 'N/A'}</p>
              <p><strong>Calibration:</strong> ${analysisResults.angle_scale_factor ? 'Dynamic overlay calibration applied' : 'Standard calibration'}</p>
              <p><strong>Measurement Accuracy:</strong> ${analysisResults.angle_scale_factor ? '±' + (1/analysisResults.angle_scale_factor).toFixed(3) + ' mm per pixel' : 'N/A'}</p>
            </div>
          </div>
          ` : ''}
          ${analysisResults.height_image ? `
          <div class="image-container">
            <h3>📏 Height Analysis</h3>
            <p class="image-description">Image used for measuring rib height</p>
            <img src="${analysisResults.height_image}" alt="Height Analysis" class="analysis-image" />
            <div class="image-details">
              <p><strong>Analysis Type:</strong> Height Measurement</p>
              <p><strong>Scale Factor:</strong> ${analysisResults.height_scale_factor ? analysisResults.height_scale_factor.toFixed(2) + ' pixels/mm' : 'N/A'}</p>
              <p><strong>Calibration:</strong> ${analysisResults.height_scale_factor ? 'Dynamic overlay calibration applied' : 'Standard calibration'}</p>
              <p><strong>Measurement Accuracy:</strong> ${analysisResults.height_scale_factor ? '±' + (1/analysisResults.height_scale_factor).toFixed(3) + ' mm per pixel' : 'N/A'}</p>
            </div>
          </div>
          ` : ''}
        </div>
        ` : ''}

        <div class="footer">
          <p>Generated on: ${currentDate}</p>
          <p>TATA Steel TMT Bar Analysis System</p>
        </div>
      </body>
      </html>
    `;
  }
} 