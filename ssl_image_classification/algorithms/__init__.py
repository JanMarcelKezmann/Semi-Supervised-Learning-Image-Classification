from .mixup import mixup, ssl_loss_mixup
from .mixmatch import mixmatch, ssl_loss_mixmatch
from .remixmatch import remixmatch, ssl_loss_remixmatch
from .fixmatch import fixmatch, ssl_loss_fixmatch
from .vat import vat, ssl_loss_vat
from .meanteacher import mean_teacher, ssl_loss_mean_teacher
from .pimodel import pi_model, ssl_loss_pi_model
from .pseudolabel import pseudo_label, ssl_loss_pseudo_label
from .ict import ict, ssl_loss_ict