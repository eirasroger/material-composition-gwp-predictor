; Inno Setup script for GHG Predictor.
; Build with:  iscc desktop_app\build\installer.iss
; Produces:    desktop_app\build\out\GHGPredictorSetup.exe
;
; Per-user install: lands in %LOCALAPPDATA%\GHGPredictor (no admin required).

#define MyAppName       "GHG Predictor"
#define MyAppExeName    "GHGPredictor.exe"
#define MyAppPublisher  "Material Composition GWP Predictor"
#define MyAppURL        "https://github.com/eirasroger/material-composition-gwp-predictor"

; Override on the iscc command line: /DMyAppVersion=1.0.0
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{7E2F4D5C-3A2B-4F1A-9D6E-2A4B7E0D9F11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases

; Per-user install — no admin, no UAC.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\GHGPredictor
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=yes

; Output location relative to this .iss file.
OutputDir=out
OutputBaseFilename=GHGPredictorSetup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\icon.ico

; Block install on anything older than Windows 10.
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Pull the entire one-folder PyInstaller dist tree into {app}.
Source: "out\GHGPredictor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
