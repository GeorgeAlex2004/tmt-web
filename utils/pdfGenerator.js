import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system';

export class PDFReportGenerator {
  constructor() {
    this.currentDate = new Date().toLocaleDateString();
    this.currentTime = new Date().toLocaleTimeString();
    this.testId = `TMT-${Date.now()}`;
  }

  // Generate HTML content for the PDF
  generateHTMLContent(data) {
    const {
      diameter,
      verdict,
      level1,
      level2,
      segmentedImageBase64,
      debugImageBase64,
      testId = this.testId,
      currentDate = this.currentDate,
      currentTime = this.currentTime
    } = data;

    const level1Results = this.formatLevel1Results(level1);
    const level2Results = this.formatLevel2Results(level2, diameter);
    const overallStatus = verdict === 'PASS' ? 'PASSED' : 'FAILED';
    const statusColor = verdict === 'PASS' ? '#4CAF50' : '#f44336';

    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>TMT Ring Test Report</title>
        <style>
          body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
          }
          .header {
            background: linear-gradient(135deg, #0A0A0A 0%, #1A1A2E 50%, #16213E 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
          }
          .logo {
            font-size: 32px;
            font-weight: 900;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
          }
          .logo-sub {
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 20px;
          }
          .report-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #00D4FF;
          }
          .report-id {
            font-size: 14px;
            opacity: 0.8;
          }
          .section {
            background: white;
            margin-bottom: 20px;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          }
          .section-title {
            font-size: 20px;
            font-weight: bold;
            color: #0A0A0A;
            margin-bottom: 20px;
            border-bottom: 2px solid #00D4FF;
            padding-bottom: 10px;
          }
          .status-badge {
            display: inline-block;
            padding: 15px 25px;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            font-size: 18px;
            margin: 20px 0;
            text-align: center;
            background: ${statusColor};
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
          }
          .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
          }
          .info-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00D4FF;
          }
          .info-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            font-weight: bold;
            margin-bottom: 5px;
          }
          .info-value {
            font-size: 16px;
            font-weight: bold;
            color: #333;
          }
          .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          }
          .results-table th {
            background: linear-gradient(135deg, #0A0A0A 0%, #1A1A2E 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
          }
          .results-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
          }
          .results-table tr:nth-child(even) {
            background: #f8f9fa;
          }
          .pass {
            color: #4CAF50;
            font-weight: bold;
          }
          .fail {
            color: #f44336;
            font-weight: bold;
          }
          .info {
            color: #2196F3;
            font-weight: bold;
          }
          .image-container {
            text-align: center;
            margin: 20px 0;
          }
          .analysis-image {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 2px solid #00D4FF;
          }
          .image-caption {
            font-size: 14px;
            color: #666;
            margin-top: 10px;
            font-style: italic;
          }
          .standards-list {
            list-style: none;
            padding: 0;
          }
          .standards-list li {
            padding: 10px 0;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
          }
          .standards-list li:before {
            content: "✓";
            color: #4CAF50;
            font-weight: bold;
            margin-right: 10px;
            font-size: 18px;
          }
          .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            font-size: 12px;
            color: #666;
          }
          .timestamp {
            font-size: 14px;
            color: #999;
            text-align: center;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
          }
        </style>
      </head>
      <body>
        <div class="header">
          <div class="logo">TATA STEEL</div>
          <div class="logo-sub">Advanced AI-Powered Analysis</div>
          <div class="report-title">TMT Bar Ring Test Report</div>
          <div class="report-id">Report ID: ${testId}</div>
        </div>

        <div class="section">
          <div class="section-title">📋 Test Information</div>
          <div class="info-grid">
            <div class="info-item">
              <div class="info-label">Test Date</div>
              <div class="info-value">${currentDate}</div>
            </div>
            <div class="info-item">
              <div class="info-label">Test Time</div>
              <div class="info-value">${currentTime}</div>
            </div>
            <div class="info-item">
              <div class="info-label">Bar Diameter</div>
              <div class="info-value">${diameter} mm</div>
            </div>
            <div class="info-item">
              <div class="info-label">Test Type</div>
              <div class="info-value">Ring Test (NITOL)</div>
            </div>
          </div>
          
          <div class="status-badge">
            🎯 Overall Result: ${overallStatus}
          </div>
        </div>

        <div class="section">
          <div class="section-title">🔬 Level 1: Color & Shape Analysis</div>
          <table class="results-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Result</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              ${level1Results.map(result => `
                <tr>
                  <td>${result.label}</td>
                  <td>${result.value}</td>
                  <td class="${result.status}">${result.statusText}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <div class="section">
          <div class="section-title">📏 Level 2: Dimensional Analysis</div>
          <table class="results-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              ${level2Results.map(result => `
                <tr>
                  <td>${result.label}</td>
                  <td>${result.value}</td>
                  <td class="${result.status}">${result.statusText}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        ${segmentedImageBase64 ? `
        <div class="section">
          <div class="section-title">🖼️ Level 1: Segmented Image Analysis</div>
          <div class="image-container">
            <img src="data:image/png;base64,${segmentedImageBase64}" class="analysis-image" alt="Segmented Image" />
            <div class="image-caption">Cross-section analysis after NITOL application</div>
          </div>
        </div>
        ` : ''}

        ${debugImageBase64 ? `
        <div class="section">
          <div class="section-title">📐 Level 2: Thickness Analysis</div>
          <div class="image-container">
            <img src="data:image/png;base64,${debugImageBase64}" class="analysis-image" alt="Thickness Analysis" />
            <div class="image-caption">Rim thickness measurement and analysis</div>
          </div>
        </div>
        ` : ''}

        <div class="section">
          <div class="section-title">📋 Quality Standards Reference</div>
          <ul class="standards-list">
            <li>Rim thickness should be 7-10% of bar diameter</li>
            <li>Three distinct layers: Rim (Dark), Transition, Core (Light)</li>
            <li>Continuous outer ring structure</li>
            <li>Concentric regions with uniform thickness</li>
            <li>Proper color contrast after NITOL application</li>
          </ul>
        </div>

        <div class="footer">
          <p><strong>TATA STEEL</strong> - TMT Ring Test Analysis System</p>
          <p>This report was generated using advanced AI-powered image analysis</p>
          <p>For technical support, contact the quality assurance team</p>
        </div>

        <div class="timestamp">
          Report generated on ${currentDate} at ${currentTime}<br>
          Test ID: ${testId}
        </div>
      </body>
      </html>
    `;
  }

  // Format Level 1 results for table
  formatLevel1Results(level1) {
    return [
      {
        label: 'Layers Detected',
        value: level1?.dark_grey_and_light_core_visible ? 'Yes' : 'No',
        status: level1?.dark_grey_and_light_core_visible ? 'pass' : 'fail',
        statusText: level1?.dark_grey_and_light_core_visible ? 'PASS' : 'FAIL'
      },
      {
        label: 'Continuous Outer Ring',
        value: level1?.continuous_outer_ring ? 'Yes' : 'No',
        status: level1?.continuous_outer_ring ? 'pass' : 'fail',
        statusText: level1?.continuous_outer_ring ? 'PASS' : 'FAIL'
      },
      {
        label: 'Concentric Regions',
        value: level1?.concentric_regions ? 'Yes' : 'No',
        status: level1?.concentric_regions ? 'pass' : 'fail',
        statusText: level1?.concentric_regions ? 'PASS' : 'FAIL'
      },
      {
        label: 'Uniform Thickness',
        value: level1?.uniform_thickness ? 'Yes' : 'No',
        status: level1?.uniform_thickness ? 'pass' : 'fail',
        statusText: level1?.uniform_thickness ? 'PASS' : 'FAIL'
      }
    ];
  }

  // Format Level 2 results for table
  formatLevel2Results(level2, diameter) {
    const minThickness = level2?.min_thickness_mm || 0;
    const maxThickness = level2?.max_thickness_mm || 0;
    const minPercentage = ((minThickness / diameter) * 100).toFixed(1);
    const maxPercentage = ((maxThickness / diameter) * 100).toFixed(1);
    const qualityStatus = level2?.quality_status || false;

    return [
      {
        label: 'Minimum Thickness',
        value: `${minThickness} mm`,
        status: 'info',
        statusText: 'MEASURED'
      },
      {
        label: 'Maximum Thickness',
        value: `${maxThickness} mm`,
        status: 'info',
        statusText: 'MEASURED'
      },
      {
        label: 'Thickness Range',
        value: `${minThickness} - ${maxThickness} mm`,
        status: 'info',
        statusText: 'MEASURED'
      },
      {
        label: 'Thickness Percentage',
        value: `${minPercentage}% - ${maxPercentage}%`,
        status: 'info',
        statusText: 'MEASURED'
      },
      {
        label: 'Within Standard Range',
        value: qualityStatus ? 'Yes' : 'No',
        status: qualityStatus ? 'pass' : 'fail',
        statusText: qualityStatus ? 'PASS' : 'FAIL'
      },
      {
        label: 'Quality Status',
        value: qualityStatus ? 'Meets Standards' : 'Below Standards',
        status: qualityStatus ? 'pass' : 'fail',
        statusText: qualityStatus ? 'PASS' : 'FAIL'
      }
    ];
  }

  // Generate and share PDF
  async generateAndSharePDF(data) {
    try {
      const htmlContent = this.generateHTMLContent(data);
      
      // Generate PDF
      const { uri } = await Print.printToFileAsync({
        html: htmlContent,
        base64: false
      });

      // Create a copy with a proper filename
      const fileName = `TMT_Report_${this.testId}.pdf`;
      const newUri = `${FileSystem.Paths.documentDirectory}${fileName}`;
      
      await FileSystem.moveAsync({
        from: uri,
        to: newUri
      });

      // Share the PDF
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(newUri, {
          mimeType: 'application/pdf',
          dialogTitle: 'TMT Ring Test Report'
        });
      }

      return newUri;
    } catch (error) {
      console.error('PDF generation error:', error);
      throw new Error('Failed to generate PDF report');
    }
  }

  // Generate PDF without sharing (for preview)
  async generatePDF(data) {
    try {
      const htmlContent = this.generateHTMLContent(data);
      
      const { uri } = await Print.printToFileAsync({
        html: htmlContent,
        base64: false
      });

      return uri;
    } catch (error) {
      console.error('PDF generation error:', error);
      throw new Error('Failed to generate PDF report');
    }
  }
}

export default PDFReportGenerator; 