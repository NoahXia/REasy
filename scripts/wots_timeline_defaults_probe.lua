-- WotS passive event default probe. No setters, allocations or skipped game calls.
local TYPE_NAMES = {
    "app.motion_track.AIControl",
    "app.motion_track.AddMotionPlayerBlendRateTrack",
    "app.motion_track.AfterImageTrack",
    "app.motion_track.AppPadVibrationTrack",
    "app.motion_track.AttackCollision",
    "app.motion_track.AttackCollision_Body",
    "app.motion_track.AttackCollision_Gimmick",
    "app.motion_track.AttackCollision_Parryable",
    "app.motion_track.AttackCollision_Wp",
    "app.motion_track.AttackInformationTrack",
    "app.motion_track.BeltAdjustTrack",
    "app.motion_track.BloodSplatter",
    "app.motion_track.BloodTrailTrack",
    "app.motion_track.CLSPVirtualGroundDisableTrack",
    "app.motion_track.CameraDepthOfField",
    "app.motion_track.CameraEvent",
    "app.motion_track.CancelPhase",
    "app.motion_track.ChangeModelPartsCondition",
    "app.motion_track.CharacterActionState",
    "app.motion_track.CharacterAimTarget",
    "app.motion_track.CharacterAllMudSpTypeTrack",
    "app.motion_track.CharacterAnimationBasic",
    "app.motion_track.CharacterAxisXTargetAim",
    "app.motion_track.CharacterBasic",
    "app.motion_track.CharacterCallEffectInteractGimmick",
    "app.motion_track.CharacterChainBlend",
    "app.motion_track.CharacterChangeCollisionEnable_Wp",
    "app.motion_track.CharacterColliderShapeTransform",
    "app.motion_track.CharacterDirectDamageTrack",
    "app.motion_track.CharacterDisableCollision",
    "app.motion_track.CharacterEdgeHandCatch",
    "app.motion_track.CharacterGrappleDisableTerrainAdjust",
    "app.motion_track.CharacterGrappleTilt",
    "app.motion_track.CharacterGrappleTrack",
    "app.motion_track.CharacterHitStopTrack",
    "app.motion_track.CharacterInvincibleTrackBase",
    "app.motion_track.CharacterMarkerPositionTrack",
    "app.motion_track.CharacterMeshCutTrack",
    "app.motion_track.CharacterNoHitTrack",
    "app.motion_track.CharacterOverwritePressLevel",
    "app.motion_track.CharacterRequestChanbaraMsg",
    "app.motion_track.CharacterRequestStageGimmick",
    "app.motion_track.CharacterRikidoTrack",
    "app.motion_track.CharacterSoulBoostTrack",
    "app.motion_track.CharacterStanceDetailTrack",
    "app.motion_track.CharacterSuperArmorTrack",
    "app.motion_track.CharacterWeaponTrack",
    "app.motion_track.CharacterWorkRateChangeTrack",
    "app.motion_track.DamageCollision",
    "app.motion_track.DynamicsForceDisableOnMeshCutTrack",
    "app.motion_track.Em101AttackSlow",
    "app.motion_track.Em102GroundState",
    "app.motion_track.Em102SwellState",
    "app.motion_track.Em104ApplySoulBoostMaterialTrack",
    "app.motion_track.Em104EnableRollingAtkVfxTrack",
    "app.motion_track.Em105ApplyModelPartsConditionTrack",
    "app.motion_track.Em112ChanbaraSpeakTrack",
    "app.motion_track.Em300ChanbaraSpeakTrack",
    "app.motion_track.Em300PowerUpTrack",
    "app.motion_track.Em503BasicTrack",
    "app.motion_track.Em503ChildMotionEnd",
    "app.motion_track.Em503ChildMotionStart",
    "app.motion_track.Em503ChildSubUnitIdlePointTypeChange",
    "app.motion_track.Em503ChildSubUnitRequestEnhanceEffectTrack",
    "app.motion_track.Em503ChildSubUnitSmoothTimeScale",
    "app.motion_track.Em503SoulBoostStabSwordChangeTrack",
    "app.motion_track.Em503SubUnitControlFlowLightMaterial",
    "app.motion_track.Em503SubUnitFaceMotTrack",
    "app.motion_track.Em503SubUnitLockOnEnable",
    "app.motion_track.Em503TentacleMotionSpeed",
    "app.motion_track.Em503TentacleParryCollisionSwitch",
    "app.motion_track.EnemyAIControl",
    "app.motion_track.EnemyActionAcception",
    "app.motion_track.EnemyActionAttribute",
    "app.motion_track.EnemyActionPerformancePoseTrack",
    "app.motion_track.EnemyActionTransitionTrack",
    "app.motion_track.EnemyAtemiTrack",
    "app.motion_track.EnemyAttackAdjustment",
    "app.motion_track.EnemyAttackGuideTrack",
    "app.motion_track.EnemyBasic",
    "app.motion_track.EnemyBossDeathCameraTrack",
    "app.motion_track.EnemyCameraEvent",
    "app.motion_track.EnemyCheckTerrainTrack",
    "app.motion_track.EnemyCombatEmotionTrack",
    "app.motion_track.EnemyConditionStatusTrack",
    "app.motion_track.EnemyCounterIssenAcceptTrack",
    "app.motion_track.EnemyCounterStanceSlowTrack",
    "app.motion_track.EnemyDeactiveFireCollisionTrack",
    "app.motion_track.EnemyDieFadeoutTrack",
    "app.motion_track.EnemyDieMeshCutTrack",
    "app.motion_track.EnemyEnableHitDummyAttack",
    "app.motion_track.EnemyFinishFireVfxTrack",
    "app.motion_track.EnemyFireDamagedTrack",
    "app.motion_track.EnemyForceLockonSuspend",
    "app.motion_track.EnemyForcePlayerCommandMaskTrack",
    "app.motion_track.EnemyJumpMoveTrack",
    "app.motion_track.EnemyJustGuardActionChangeTrack",
    "app.motion_track.EnemyLadderTrack",
    "app.motion_track.EnemyNoHitTrack",
    "app.motion_track.EnemyNoPartsDamageTrack",
    "app.motion_track.EnemyNoRikidoDamageTrack",
    "app.motion_track.EnemyParryTargetSearch",
    "app.motion_track.EnemyParryTurnTarget",
    "app.motion_track.EnemyPartsEnableAdjustParamTrack",
    "app.motion_track.EnemyPartsEnableTrack",
    "app.motion_track.EnemyPhaseChangeTrack",
    "app.motion_track.EnemyRequestStageGimmick",
    "app.motion_track.EnemyRikidoIssenActivateTrack",
    "app.motion_track.EnemyRotateTargetObject",
    "app.motion_track.EnemySafetyObstacleBreakTrack",
    "app.motion_track.EnemyShout",
    "app.motion_track.EnemySmashUpperAdjustTrack",
    "app.motion_track.EnemySoulAbsorption",
    "app.motion_track.EnemySoulAmountTrack",
    "app.motion_track.EnemyStepTrack",
    "app.motion_track.EnemySuperArmorTrack",
    "app.motion_track.EnemyTargetChangeableMove",
    "app.motion_track.EnemyTiredAddBlend",
    "app.motion_track.EnemyUniqueReactionState",
    "app.motion_track.EnemyWarpAtCogTrack",
    "app.motion_track.EnvUnitBasicMeshPartsChange",
    "app.motion_track.Eu001MeshPartsChange",
    "app.motion_track.Eu015MeshPartsChange",
    "app.motion_track.FacialLipBlendRate",
    "app.motion_track.FacialLipMotionSwitch",
    "app.motion_track.FacialMotionSet",
    "app.motion_track.FullBodyShakeBlendRate",
    "app.motion_track.GimmickBasic",
    "app.motion_track.GimmickRemoveShell",
    "app.motion_track.Gm007EnabledRidigBody",
    "app.motion_track.Gm049EnabledAttackCollision",
    "app.motion_track.Gm049EnabledRidigBody",
    "app.motion_track.GrappleRequestNextFlowTrack",
    "app.motion_track.HandFix",
    "app.motion_track.HitVfxOverwriteTrack",
    "app.motion_track.HumanoidBoneBasic",
    "app.motion_track.IKLegCharacterTilt2Leg",
    "app.motion_track.IKLegCharacterTilt4Leg",
    "app.motion_track.IkArm",
    "app.motion_track.IkLegBlendRate",
    "app.motion_track.IkLegTilt",
    "app.motion_track.JointConstraintsLayer",
    "app.motion_track.JointOffsetController",
    "app.motion_track.LWeaponConst",
    "app.motion_track.LongObjectControlTrack",
    "app.motion_track.LookAtBlendRate",
    "app.motion_track.LookAtParameterSetSelector",
    "app.motion_track.MotionTransLeverAdjustTrack",
    "app.motion_track.NoiseRequestTrack",
    "app.motion_track.OniGateBasic",
    "app.motion_track.PlayerActionTransitionTrack",
    "app.motion_track.PlayerAutoWeaponOnOffTrack",
    "app.motion_track.PlayerBasic",
    "app.motion_track.PlayerBasicSubLayer",
    "app.motion_track.PlayerChainIssenTrack",
    "app.motion_track.PlayerCloakControlTrack",
    "app.motion_track.PlayerCommandCancel",
    "app.motion_track.PlayerCounterIssenAdjustTrack",
    "app.motion_track.PlayerInputAdjust",
    "app.motion_track.PlayerInputRotate",
    "app.motion_track.PlayerInternalLockonTrack",
    "app.motion_track.PlayerIssenAttackDirect",
    "app.motion_track.PlayerMoveBank",
    "app.motion_track.PlayerNoHitTrack",
    "app.motion_track.PlayerNoticeTrack",
    "app.motion_track.PlayerOniChange",
    "app.motion_track.PlayerPadVibration",
    "app.motion_track.PlayerSkillTrack",
    "app.motion_track.PlayerSoulBoostSpawnShell",
    "app.motion_track.PlayerSoulBoostTrack",
    "app.motion_track.PlayerSubWeaponTrack",
    "app.motion_track.PlayerSuperArmorTrack",
    "app.motion_track.PlayerTrackingTarget",
    "app.motion_track.PressCollision",
    "app.motion_track.RWeaponConst",
    "app.motion_track.RagdollControl",
    "app.motion_track.ResetTentacleBloodValueTrack",
    "app.motion_track.RikidoTiredBlendTrack",
    "app.motion_track.SensorCollision",
    "app.motion_track.SheathHold",
    "app.motion_track.SoulAbsorption",
    "app.motion_track.SoulAbsorptionSub",
    "app.motion_track.SoulBoostTrailVfxTrack",
    "app.motion_track.SoulBoostVfxControlTrack",
    "app.motion_track.SoulGenerate",
    "app.motion_track.SpawnMultiShell",
    "app.motion_track.SpawnShell",
    "app.motion_track.SpawnShellLaser",
    "app.motion_track.SpawnShellRandom",
    "app.motion_track.StageVfxTrack",
    "app.motion_track.VFXBeginIdleEndRangeTrack",
    "app.motion_track.VFXTrigger_Wp",
    "app.motion_track.VfxDrawOffTrack",
    "app.motion_track.WeaponBloodWetTrack",
    "app.motion_track.WeaponGroundAdjust",
    "app.motion_track.WeaponSpawnShell",
    "app.motion_track.WorkRateTrack",
    "app.motion_track.cCancelCommand",
    "app.motion_track.cCancelCommandContinueInputTransition",
    "app.motion_track.cCancelCommandFeintAttack",
    "app.motion_track.cSoulGenerateOverwriteParam",
    "app.motion_track.cVfxIdSelector",
    "app.motion_track.cVfxOverwriteParam",
}

if _G.WOTS_TIMELINE_DEFAULTS_PROBE then return end
local S={version='1.0',recording=false,installed=false,frame=0,events={},errors={},hooks={},counts={},seen={},last_sample={},per_frame=0,dropped=0,status='Hooks off'}
_G.WOTS_TIMELINE_DEFAULTS_PROBE=S
local allowed={};for _,n in ipairs(TYPE_NAMES) do allowed[n]=true end
local function safe(fn,fallback) local ok,v=pcall(fn);if ok then return v end;return fallback end
local function err(v)
    if #S.errors<100 then S.errors[#S.errors+1]=tostring(v) end
    if #S.errors>=30 then S.recording=false;S.status='Paused after read errors; export for diagnosis' end
end
local function managed(raw)
    if raw==nil then return nil end
    local bits=safe(function() return sdk.to_int64(raw) end,0)
    if type(bits)~='number' or bits<0x100000 or bits%8~=0 then return nil end
    if not safe(function() return sdk.is_managed_object(raw) end,false) then return nil end
    return safe(function() return sdk.to_managed_object(raw) end,nil)
end
local scalar={['System.Boolean']=true,['System.Single']=true,['System.Double']=true,
 ['System.Int32']=true,['System.UInt32']=true,['System.Int16']=true,['System.UInt16']=true,
 ['System.SByte']=true,['System.Byte']=true}
local function snapshot(obj,depth,visited)
    local td=obj:get_type_definition();local typename=td:get_full_name()
    local address=tostring(obj:get_address())
    local result={type=typename,fields={}}
    if visited[address] or depth<=0 then return {type=typename,unresolved='depth/cycle'} end
    visited[address]=true
    local cursor=td;local count=0
    while cursor and count<160 do
        for _,f in ipairs(cursor:get_fields() or {}) do
            local n=f:get_name()
            if not f:is_static() and result.fields[n]==nil then
                count=count+1
                local ft=f:get_type();local fn=ft:get_full_name();local value
                if scalar[fn] then
                    value=safe(function() return f:get_data(obj) end,{unresolved='field read'})
                    if type(value)=='number' and (value~=value or math.abs(value)==math.huge) then value=tostring(value) end
                elseif fn=='System.Int64' or fn=='System.UInt64' then
                    value={unresolved='64-bit scalar; retained as raw hex',raw=safe(function() return tostring(obj:read_qword(f:get_offset_from_base())) end,'read failed')}
                elseif safe(function() return ft:is_enum() end,false) then
                    value={unresolved='enum underlying representation not verified'}
                elseif not safe(function() return ft:is_value_type() end,true) then
                    local off=f:get_offset_from_base()
                    local raw=safe(function() return obj:read_qword(off) end,nil)
                    if raw==0 then value={is_null=true}
                    else
                        local child=raw and managed(sdk.to_ptr(raw)) or nil
                        if child and (fn:find('app.',1,true)==1 or fn:find('ace.',1,true)==1) then
                            value=snapshot(child,depth-1,visited)
                        else value={unresolved='reference not traversed'} end
                    end
                else value={unresolved='value type not decoded'} end
                result.fields[n]={declared_type=fn,value=value}
            end
        end
        cursor=cursor:get_parent_type()
    end
    if count>=160 then result.truncated=true end
    visited[address]=nil
    return result
end
local function capture(raw,stage)
    if not S.recording then return end
    local obj=managed(raw);if not obj then return end
    local name=obj:get_type_definition():get_full_name();if not allowed[name] then return end
    local key=name..':'..stage
    if (S.counts[key] or 0)>=12 then return end
    if S.per_frame>=12 then S.dropped=S.dropped+1;return end
    if stage~='constructor_after' and S.last_sample[key] and S.frame-S.last_sample[key]<30 then return end
    S.last_sample[key]=S.frame;S.per_frame=S.per_frame+1
    if #S.events>=3000 then S.recording=false;S.status='Sample limit reached; export now';return end
    local shot=snapshot(obj,3,{})
    local fingerprint=json.dump_string(shot)
    S.seen[key]=S.seen[key] or {}
    if S.seen[key][fingerprint] then return end
    S.seen[key][fingerprint]=true;S.counts[key]=(S.counts[key] or 0)+1
    S.events[#S.events+1]={frame=S.frame,stage=stage,type=name,values=shot}
end
local function install()
    if S.installed then return end
    if not thread or not thread.get_hook_storage then S.status='Missing hook storage support';return end
    local groups={}
    for _,n in ipairs(TYPE_NAMES) do
        local td=sdk.find_type_definition(n)
        if td then
            for _,m in ipairs(td:get_methods() or {}) do
                local mn=m:get_name()
                if (mn=='.ctor' or mn=='resetEveryFrame') and not m:is_static()
                    and #(m:get_param_types() or {})==0 and m:get_declaring_type():get_full_name()==n then
                    local addr=tostring(m:get_function());local stage=mn=='.ctor' and 'constructor' or 'reset'
                    if not groups[addr] then groups[addr]={method=m,stage=stage,names={n},address=addr}
                    else
                        local g=groups[addr];g.names[#g.names+1]=n
                        if g.stage~=stage then g.ambiguous=true end
                    end
                end
            end
        end
    end
    for _,g in pairs(groups) do
        local row={stage=g.stage,types=g.names,address=g.address};S.hooks[#S.hooks+1]=row
        if g.ambiguous then row.status='skipped: shared constructor/reset implementation'
        else
            local ok,why=pcall(function()
                sdk.hook(g.method,function(args)
                    local storage=thread.get_hook_storage();storage.timeline_probe_raw=nil
                    if not S.recording then return end
                    storage.timeline_probe_raw=args[2]
                    if g.stage=='reset' then local good,e=pcall(capture,args[2],'reset_before');if not good then err(e) end end
                end,function(retval)
                    local raw=thread.get_hook_storage().timeline_probe_raw
                    if raw then local good,e=pcall(capture,raw,g.stage..'_after');if not good then err(e) end end
                    return retval
                end,true)
            end)
            row.status=ok and 'installed' or 'failed';if not ok then err(why) end
        end
    end
    S.installed=true;S.status='Hooks installed; capture off'
end
local function export()
    S.recording=false
    local path='wots_timeline_defaults_'..os.date('%Y%m%d_%H%M%S')..'.json'
    local ok,v=pcall(json.dump_file,path,{version=S.version,tdb=sdk.get_tdb_version(),
        scope='Passive natural constructors and resetEveryFrame only; reset_before is NOT a default or confirmed dispatch sample.',
        limits={per_type_stage=12,total=3000,depth=3},events=S.events,hooks=S.hooks,errors=S.errors,counts=S.counts,dropped=S.dropped})
    S.status=ok and v~=false and ('Exported: '..path) or ('Export failed: '..tostring(v))
end
re.on_frame(function() S.frame=S.frame+1;S.per_frame=0 end)
re.on_draw_ui(function()
    if not imgui.tree_node('WOTS Timeline Defaults Probe') then return end
    imgui.text(S.status);imgui.text('Samples: '..#S.events..' | Errors: '..#S.errors)
    imgui.text('Passive capture. Constructor/reset samples remain separate. No game values are changed.')
    if not S.installed and imgui.button('Install timeline hooks') then local ok,e=pcall(install);if not ok then err(e);S.status=tostring(e) end end
    if S.installed and not S.recording and imgui.button('Start capture') then S.recording=true;S.status='Capturing; play normally, then Stop and export' end
    if imgui.button('Stop and export JSON') then export() end
    imgui.tree_pop()
end)
