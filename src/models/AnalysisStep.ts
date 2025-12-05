export enum AnalysisStep {
  ANGLE = 'angle',
  HEIGHT = 'height',
}

export interface AnalysisStepInfo {
  title: string;
  description: string;
  stepNumber: number;
}

export const getAnalysisStepInfo = (step: AnalysisStep): AnalysisStepInfo => {
  switch (step) {
    case AnalysisStep.ANGLE:
      return {
        title: 'Front View (Rib Angle)',
        description: 'Capture photo of TMT bar from front view to measure rib angle',
        stepNumber: 1,
      };
    case AnalysisStep.HEIGHT:
      return {
        title: 'Height View (45° Angle)',
        description: 'Capture photo of TMT bar at 45° angle to measure rib height',
        stepNumber: 2,
      };
    default:
      throw new Error(`Unknown analysis step: ${step}`);
  }
}; 