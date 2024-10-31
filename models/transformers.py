from diffusers import CogVideoXTransformer3DModel

class CogVideoXTransformer3DModelPose(CogVideoXTransformer3DModel):
    def __init__(self, **kwargs):
        super(CogVideoXTransformer3DModelPose, self).__init__(**kwargs)

    def from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        # TODO
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # pose_emb
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    )