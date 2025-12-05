export class AnalysisCalculator {
  static calculateRibAngle({
    calibrationSize,
    measuredSize,
    barDiameter,
  }: {
    calibrationSize: number;
    measuredSize: number;
    barDiameter: number;
  }): number {
    // Convert measurements to radians and calculate angle
    const angleRadians = Math.atan2(measuredSize, calibrationSize);
    return Math.abs((angleRadians * 180) / Math.PI);
  }

  static calculateRibHeight({
    calibrationSize,
    measuredSize,
    barDiameter,
  }: {
    calibrationSize: number;
    measuredSize: number;
    barDiameter: number;
  }): number {
    // Calculate height based on calibration and measured size
    const height = (measuredSize / calibrationSize) * barDiameter;
    return parseFloat(height.toFixed(2));
  }

  static calculateARValue({
    ribAngle,
    ribHeight,
    barDiameter,
  }: {
    ribAngle: number;
    ribHeight: number;
    barDiameter: number;
  }): number {
    // Convert angle to radians for calculation
    const angleRadians = (ribAngle * Math.PI) / 180;
    
    // Calculate AR value using the formula:
    // AR = (2 * h * sin(θ)) / (π * d)
    // where h is rib height, θ is rib angle, and d is bar diameter
    const arValue = (2 * ribHeight * Math.sin(angleRadians)) / (Math.PI * barDiameter);
    return parseFloat(arValue.toFixed(3));
  }

  static validateMeasurements({
    angle,
    height,
    barDiameter,
  }: {
    angle: number;
    height: number;
    barDiameter: number;
  }): Record<string, string> {
    const errors: Record<string, string> = {};

    // Validate angle (should be between 30 and 60 degrees)
    if (angle < 30 || angle > 60) {
      errors['angle'] = 'Rib angle should be between 30° and 60°';
    }

    // Validate height (should be between 0.4 and 0.6 times the bar diameter)
    const minHeight = 0.4 * barDiameter;
    const maxHeight = 0.6 * barDiameter;
    if (height < minHeight || height > maxHeight) {
      errors['height'] = `Rib height should be between ${minHeight.toFixed(1)}mm and ${maxHeight.toFixed(1)}mm`;
    }

    return errors;
  }
} 