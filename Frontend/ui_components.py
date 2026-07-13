# ============================================================================
# UI COMPONENTS FOR BACKEND INTEGRATION
# ============================================================================
"""
Kivy widgets for displaying:
- System mode (IoT/API/Hybrid)
- Connection status
- Predictions and confidence
- Error messages
- Auto-refresh status
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard

import logging

logger = logging.getLogger(__name__)


class ModeIndicatorWidget(MDLabel):
    """Displays current system mode with color coding."""
    
    mode = StringProperty("api")  # iot, api, hybrid
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(mode=self._update_display)
        self._update_display()
    
    def _update_display(self, *args):
        """Update display based on mode."""
        mode = self.mode
        
        if mode == "iot":
            self.text = "Live IoT Sensor"
            self.text_color = (0.18, 0.80, 0.28, 1)  # Green
        elif mode == "api":
            self.text = "External API Data"
            self.text_color = (0.98, 0.65, 0.05, 1)  # Amber
        elif mode == "hybrid":
            self.text = "IoT + API Hybrid"
            self.text_color = (0.18, 0.80, 0.28, 1)  # Green
        else:
            self.text = "Detecting source..."
            self.text_color = (0.5, 0.5, 0.5, 1)  # Gray
        
        self.font_style = "Caption"
        self.size_hint_y = None
        self.height = dp(22)
        logger.debug(f"Mode indicator updated: {mode}")


class ConnectionStatusWidget(MDLabel):
    """Displays connection status to backend."""
    
    is_connected = BooleanProperty(True)
    retry_count = NumericProperty(0)
    max_retries = NumericProperty(3)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(is_connected=self._update_display, retry_count=self._update_display)
        self._update_display()
    
    def _update_display(self, *args):
        """Update display based on connection state."""
        if self.is_connected:
            self.text = "✓ Connected"
            self.text_color = (0.18, 0.80, 0.28, 1)  # Green
        elif self.retry_count > 0:
            retry_pct = int((self.retry_count / self.max_retries) * 100)
            self.text = f"↻ Reconnecting... ({retry_pct}%)"
            self.text_color = (0.98, 0.85, 0.05, 1)  # Yellow
        else:
            self.text = "✗ Offline"
            self.text_color = (0.90, 0.18, 0.18, 1)  # Red
        
        self.font_style = "Caption"


class PredictionDisplayCard(MDCard):
    """Card displaying current prediction and confidence."""
    
    prediction_value = NumericProperty(0)
    prediction_unit = StringProperty("µg/m³")
    confidence_value = NumericProperty(0)
    source = StringProperty("API")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = "16dp"
        self.spacing = "8dp"
        self.size_hint_y = None
        self.height = "120dp"
        self.md_bg_color = (0.96, 0.97, 0.99, 1)
        
        # Title
        title = MDLabel(
            text="ML Model Prediction",
            font_style="Subtitle1",
            bold=True,
            theme_text_color="Custom",
            text_color=(0.08, 0.12, 0.22, 1),
            size_hint_y=None,
            height="24dp"
        )
        self.add_widget(title)
        
        # Prediction value
        self.pred_label = MDLabel(
            text="--",
            font_style="H4",
            bold=True,
            theme_text_color="Custom",
            text_color=(0.00, 0.69, 0.64, 1),
            size_hint_y=None,
            height="36dp"
        )
        self.add_widget(self.pred_label)
        
        # Confidence and source
        self.conf_label = MDLabel(
            text="Confidence: --%  |  Source: --",
            font_style="Caption",
            theme_text_color="Custom",
            text_color=(0.42, 0.45, 0.52, 1),
            size_hint_y=None,
            height="16dp"
        )
        self.add_widget(self.conf_label)
        
        # Bind updates
        self.bind(
            prediction_value=self._update_display,
            confidence_value=self._update_display,
            source=self._update_display
        )
    
    def _update_display(self, *args):
        """Update display values."""
        self.pred_label.text = f"{self.prediction_value:.1f} {self.prediction_unit}"
        self.conf_label.text = (
            f"Confidence: {self.confidence_value:.0%}  |  Source: {self.source}"
        )


class ErrorMessageWidget(MDLabel):
    """Displays error messages with auto-clear."""
    
    error_text = StringProperty("")
    auto_clear_time = NumericProperty(5)  # seconds
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_text_color = "Custom"
        self.text_color = (0.90, 0.18, 0.18, 1)  # Red
        self.font_style = "Caption"
        self.bind(error_text=self._on_error)
    
    def _on_error(self, *args):
        """Handle error display with auto-clear."""
        self.text = self.error_text
        
        if self.error_text:
            # Schedule auto-clear
            Clock.schedule_once(
                lambda dt: setattr(self, 'error_text', ''),
                self.auto_clear_time
            )
    
    def show_error(self, message: str, duration: int = None):
        """Show error with optional custom duration."""
        if duration:
            self.auto_clear_time = duration
        self.error_text = message


class RefreshStatusWidget(MDLabel):
    """Shows auto-refresh activity indicator."""
    
    is_refreshing = BooleanProperty(False)
    last_refresh = StringProperty("Never")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_text_color = "Custom"
        self.text_color = (0.42, 0.45, 0.52, 1)
        self.font_style = "Overline"
        self.bind(is_refreshing=self._update_display)
    
    def _update_display(self, *args):
        """Update refresh indicator."""
        if self.is_refreshing:
            self.text = f"↻ Refreshing... (Last: {self.last_refresh})"
        else:
            self.text = f"Last updated: {self.last_refresh}"


class HistoryDisplayWidget(BoxLayout):
    """Displays recent prediction history in a compact format."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = "8dp"
        self.spacing = "4dp"
    
    def update_history(self, history_data: list):
        """
        Update with history data.
        
        Args:
            history_data: List of prediction records
        """
        self.clear_widgets()
        
        if not history_data:
            empty_label = MDLabel(
                text="No history available",
                theme_text_color="Custom",
                text_color=(0.42, 0.45, 0.52, 1),
                font_style="Caption"
            )
            self.add_widget(empty_label)
            return
        
        # Show last 5 records
        for i, record in enumerate(history_data[:5]):
            timestamp = record.get('timestamp', 'Unknown')
            prediction = record.get('prediction', 'N/A')
            
            record_label = MDLabel(
                text=f"{timestamp}: {prediction:.1f} µg/m³",
                theme_text_color="Custom",
                text_color=(0.08, 0.12, 0.22, 1),
                font_style="Caption",
                size_hint_y=None,
                height="16dp"
            )
            self.add_widget(record_label)


# ─────────────────────────────────────────────────────────────────────────
# BUILDER STRINGS (for .kv files)
# ─────────────────────────────────────────────────────────────────────────

"""
Add these to your dashboard.kv or profile.kv to use the widgets:

<ModeIndicatorWidget>:
    size_hint_x: 1
    size_hint_y: None
    height: "28dp"

<ConnectionStatusWidget>:
    size_hint_x: 1
    size_hint_y: None
    height: "20dp"

<PredictionDisplayCard>:
    pass

<ErrorMessageWidget>:
    size_hint_x: 1
    size_hint_y: None
    height: "16dp"
    
<RefreshStatusWidget>:
    size_hint_x: 1
    size_hint_y: None
    height: "16dp"

<HistoryDisplayWidget>:
    size_hint_x: 1
    size_hint_y: None
    height: "100dp"

# Usage in Dashboard:

MDBoxLayout:
    orientation: "vertical"
    padding: "16dp"
    spacing: "8dp"
    
    # Mode and connection status
    ModeIndicatorWidget:
        id: mode_indicator
        mode: "api"
    
    ConnectionStatusWidget:
        id: connection_status
        is_connected: True
    
    # Prediction card
    PredictionDisplayCard:
        id: prediction_card
        prediction_value: 85.5
        confidence_value: 0.92
        source: "GRU+SARIMA"
    
    # Error display
    ErrorMessageWidget:
        id: error_display
    
    # Refresh status
    RefreshStatusWidget:
        id: refresh_status
        last_refresh: "2 min ago"
    
    # History
    HistoryDisplayWidget:
        id: history_display
"""
